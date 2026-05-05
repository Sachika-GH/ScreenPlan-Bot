#!/usr/bin/env python3
"""
ScreenPlan QQ Bot — Lightweight AI Agent via OneBot v11 WebSocket.

Acts as a WebSocket SERVER that NapCatQQ connects TO (reverse WS).
Receives QQ messages, uses DeepSeek function-calling for admin + group queries.

Capabilities:
  • ScreenPlan admin — usage summaries, timelines, user lists, server status
  • Group member lookup — identify who sent what, list members
  • General AI chat — knowledge, coding, math, conversation

Dependencies (pip install):
    websockets
    httpx
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import date

import httpx
import websockets
from websockets.asyncio.server import serve

# ─── Config (env vars) ────────────────────────────────────

SPBOT_LISTEN_HOST = os.environ.get("SPBOT_LISTEN_HOST", "0.0.0.0")
SPBOT_LISTEN_PORT = int(os.environ.get("SPBOT_LISTEN_PORT", "3001"))
SPBOT_COMMAND_PREFIX = os.environ.get("SPBOT_PREFIX", "/")
SPBOT_ADMIN_TOKEN = os.environ.get("SPBOT_ADMIN_TOKEN", "")
SCREENPLAN_HOST = os.environ.get("SPBOT_SCREENPLAN_HOST", "http://localhost:5051")
SCREENPLAN_API = f"{SCREENPLAN_HOST}/api/admin"
LLM_API_KEY = os.environ.get("SPBOT_LLM_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
LLM_API_BASE = os.environ.get("SPBOT_LLM_BASE", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("SPBOT_LLM_MODEL", "deepseek-chat")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("spbot")

# ─── Conversation History ─────────────────────────────────

MAX_HISTORY_ROUNDS = 15
HISTORY_EXPIRE_SECONDS = 1800
_history: dict[str, list[dict]] = defaultdict(list)
_last_active: dict[str, float] = {}


def _chat_key(msg: dict) -> str:
    mt = msg.get("message_type", "private")
    if mt == "group":
        return f"g_{msg.get('group_id', '0')}"
    uid = msg.get("user_id") or msg.get("sender", {}).get("user_id", "0")
    return f"p_{uid}"


def _expire_old(ck: str):
    now = time.time()
    if ck in _last_active and (now - _last_active[ck]) > HISTORY_EXPIRE_SECONDS:
        _history.pop(ck, None)
        _last_active.pop(ck, None)


def _trim_history(ck: str):
    h = _history[ck]
    max_msgs = MAX_HISTORY_ROUNDS * 2
    if len(h) > max_msgs:
        _history[ck] = h[-max_msgs:]


# ─── Global state for request context ─────────────────────

_current_msg: dict = {}  # the message currently being processed


# ─── DeepSeek Function Definitions ────────────────────────

FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_user_list",
            "description": "列出 ScreenPlan 所有注册用户（ID、邮箱、显示名称）。仅用于查 ScreenPlan 数据，不要用于查 QQ 群成员身份。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_usage",
            "description": "查 ScreenPlan 用户的屏幕使用摘要（总时长、学习/娱乐占比、各设备详情）。仅 ScreenPlan 数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "ScreenPlan 用户ID"},
                    "date": {"type": "string", "description": "日期 YYYY-MM-DD，默认今天"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_timeline",
            "description": "查 ScreenPlan 用户的时间线详情（各设备活动事件）。仅 ScreenPlan 数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "ScreenPlan 用户ID"},
                    "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_server_status",
            "description": "查 ScreenPlan 服务器健康状态（运行时间、用户数、设备数）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_group_member",
            "description": "在 QQ 群中按 QQ 号或昵称/群名片查找群成员身份。用于回答「群里xxx是谁」「xxx是什么身份」等。注意：返回群成员的真实 QQ 昵称和群名片，不是 ScreenPlan 用户名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "qq_number": {"type": "string", "description": "要查找的 QQ 号（纯数字），如 2903475069"},
                    "nickname": {"type": "string", "description": "要查找的昵称或群名片关键字"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_group_members",
            "description": "列出当前 QQ 群的所有成员（QQ号、昵称、群名片）。仅用于群内查询。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_help",
            "description": "显示机器人的功能列表和使用帮助。当用户问「你能做什么」「帮助」「help」「功能」时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = """你是一个部署在 QQ 群中的 AI 助手，同时挂了 ScreenPlan 屏幕时间追踪系统。

━━━━ 你的能力域 ━━━━

你拥有三个互相独立的数据域，绝不能混淆：

① 通用 AI 知识 —— 编程、数学、常识、翻译、闲聊等。直接用自己的知识回答。
② ScreenPlan 数据 —— 通过函数查询注册用户的屏幕使用情况。只有明确跟屏幕时间/使用情况相关才调用。
③ QQ 群成员数据 —— 通过函数查询当前 QQ 群的成员列表和身份信息。只有明确问群成员身份时才调用。

━━━━ 函数调用指南 ━━━━

ScreenPlan 相关 (query_user_list / query_user_usage / query_user_timeline / query_server_status)：
- 消息提到"屏幕时间""使用情况""用了多久""时间线""设备""服务器""ScreenPlan""用户列表""注册用户"时调用
- 不要用来查 QQ 群成员！
- 不要用来回答「xxx是谁」「这个QQ号对应谁」之类的问题！

QQ 群成员相关 (query_group_member / list_group_members)：
- 消息提到「群里xxx是谁」「这个QQ号」「查一下群成员」「群里有谁」时调用
- query_group_member 可通过 QQ 号或昵称关键字查找

帮助 (show_help)：
- 消息是「帮助」「help」「你能做什么」「功能」时调用

━━━━ 严禁规则 ━━━━

1. 绝不把 ScreenPlan 用户等同于 QQ 群成员。两者是完全独立的体系。
2. 绝不用 ScreenPlan 的 query_user_list 来回答「某个 QQ 号/群成员是谁」的问题。
3. 绝不编造数据。不知道就说不知道。如果函数返回空，如实说「未找到」。
4. 如果用户在私聊中问群成员相关的问题，礼貌告知「这个功能需要在群里使用」。

━━━━ 输出格式 ━━━━

- QQ 不支持 Markdown。严禁使用 **、| 表格 |、### 标题
- 纯文本 + 缩进分行，每项独立一行
- 时长用「小时 分钟」如 "5h33min"
- 保持简洁，数据回复 ≤ 15 行"""


# ─── OneBot API helper ────────────────────────────────────

_ob_req_id = 0  # echo counter for OneBot API calls


async def onebot_api(action: str, params: dict) -> dict:
    """Call NapCatQQ internal OneBot API via the WebSocket connection.
    
    Sends an action request and waits for the response with matching echo.
    """
    global _ob_req_id
    if not _ws_clients:
        return {"error": "No NapCatQQ connection"}
    ws = next(iter(_ws_clients.values()))
    _ob_req_id += 1
    req_id = _ob_req_id
    payload = json.dumps({
        "action": action,
        "params": params,
        "echo": str(req_id),
    }, ensure_ascii=False)

    # Register a future to wait for the response
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _pending_ob_calls[str(req_id)] = future

    try:
        await ws.send(payload)
        # Wait with timeout
        result = await asyncio.wait_for(future, timeout=15)
        return result
    except asyncio.TimeoutError:
        _pending_ob_calls.pop(str(req_id), None)
        return {"error": "OneBot API timeout"}
    except Exception as e:
        _pending_ob_calls.pop(str(req_id), None)
        return {"error": str(e)}


_pending_ob_calls: dict[str, asyncio.Future] = {}


# ─── Group member cache ───────────────────────────────────

_group_member_cache: dict[str, tuple[float, list[dict]]] = {}
GROUP_CACHE_TTL = 300  # 5 minutes


async def _fetch_group_members(group_id: int) -> list[dict]:
    """Fetch (or retrieve from cache) the member list for a group."""
    gk = str(group_id)
    now = time.time()
    if gk in _group_member_cache:
        ts, members = _group_member_cache[gk]
        if now - ts < GROUP_CACHE_TTL:
            return members
    resp = await onebot_api("get_group_member_list", {"group_id": group_id})
    if "error" in resp or resp.get("status") != "ok":
        log.error(f"get_group_member_list failed: {resp}")
        return []
    members = resp.get("data", [])
    _group_member_cache[gk] = (now, members)
    return members


# ─── HTTP helpers (ScreenPlan) ────────────────────────────

def admin_api(path: str) -> dict:
    headers = {"Authorization": f"Bearer {SPBOT_ADMIN_TOKEN}"}
    try:
        r = httpx.get(f"{SCREENPLAN_API}/{path}", headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        log.error(f"Admin API error: {e}")
        return {"error": str(e)}


# ─── LLM caller ───────────────────────────────────────────

async def call_llm(messages: list, data_mode: bool = False) -> dict:
    """Call DeepSeek chat/completions. data_mode = lower temperature."""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "tools": FUNCTIONS,
        "tool_choice": "auto",
        "temperature": 0.1 if data_mode else 0.7,
        "max_tokens": 2048,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(3):
            try:
                r = await client.post(f"{LLM_API_BASE}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]
            except Exception as e:
                log.warning(f"LLM attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
        return {"content": "抱歉，AI 服务暂时不可用，请稍后重试。"}


# ─── Function executors ───────────────────────────────────

def execute_function(name: str, args: dict) -> str:
    """Execute a tool/function call. Returns result string."""

    # ── ScreenPlan functions ───────────────────────────

    if name == "query_user_list":
        data = admin_api("users")
        if "error" in data:
            return f"错误: {data['error']}"
        lines = [f"共 {data['count']} 个 ScreenPlan 注册用户："]
        for u in data.get("users", []):
            lines.append(f"  ID:{u['id']}  {u['display_name']} ({u['email']})")
        return "\n".join(lines)

    if name == "query_user_usage":
        uid = args["user_id"]
        dt = args.get("date", date.today().isoformat())
        data = admin_api(f"usage/{uid}?date={dt}")
        if "error" in data:
            return f"错误: {data['error']}"
        name = data.get("display_name", f"用户{uid}")
        total = data.get("total_minutes_all_devices", 0)
        devs = data.get("devices", [])
        lines = [f"{name}（{dt}）屏幕使用报告："]
        lines.append(f"  总使用时间：{total:.0f} 分钟（{total/60:.1f}h）")
        if data.get("overlap_minutes", 0) > 0:
            lines.append(f"  多设备重叠：{data['overlap_minutes']:.0f} 分钟")
        for d in devs:
            lines.append(
                f"  {d['device_name']}（{d['platform']}）：{d['total_minutes']:.0f}min "
                f"学习 {d['learning_pct']:.0f}% / 娱乐 {d['entertainment_pct']:.0f}%"
            )
        return "\n".join(lines)

    if name == "query_user_timeline":
        uid = args["user_id"]
        dt = args.get("date", date.today().isoformat())
        data = admin_api(f"timeline/{uid}?date={dt}")
        if "error" in data:
            return f"错误: {data['error']}"
        name = data.get("display_name", f"用户{uid}")
        devs = data.get("devices", [])
        total_events = sum(d["event_count"] for d in devs)
        lines = [f"{name}（{dt}）时间线：共 {total_events} 个活动事件"]
        for d in devs:
            lines.append(f"  {d['device_name']}（{d['platform']}）：{d['event_count']} 事件")
            for ev in d.get("events", [])[-5:]:
                ts = ev["timestamp"].split("T")[1][:5] if "T" in ev["timestamp"] else ev["timestamp"]
                lines.append(f"    {ts}  {ev['app_name']} [{ev['category']}]")
        return "\n".join(lines)

    if name == "query_server_status":
        data = admin_api("health")
        if "error" in data:
            return f"错误: {data['error']}"
        uptime_h = data.get("uptime_seconds", 0) / 3600
        return (
            f"ScreenPlan 服务器状态：\n"
            f"  状态：{data.get('status', 'unknown')}\n"
            f"  版本：v{data.get('version', '?')}\n"
            f"  运行时间：{uptime_h:.1f} 小时\n"
            f"  注册用户：{data.get('user_count', 0)} 人\n"
            f"  注册设备：{data.get('device_count', 0)} 台\n"
            f"  今日事件：{data.get('today_events', 0)} 条"
        )

    # ── Show help ─────────────────────────────────────

    if name == "show_help":
        return (
            "我能帮你做这些事：\n\n"
            "📊 ScreenPlan 屏幕管理\n"
            "  • 查询用户的屏幕使用时间\n"
            "  • 查看设备时间线\n"
            "  • 列出所有注册用户\n"
            "  • 检查服务器状态\n\n"
            "👥 QQ 群管理\n"
            "  • 按 QQ 号查群成员身份\n"
            "  • 按昵称查群成员\n"
            "  • 列出群成员列表\n\n"
            "💬 通用 AI\n"
            "  • 编程 / 数学 / 翻译 / 常识问答\n"
            "  • 闲聊与日常对话\n\n"
            "💡 在群里使用 / 开头向我提问，私聊直接说话即可"
        )

    # ── Group member functions (need async context) ───
    # These are handled in async_execute_function instead

    return f"未知函数: {name}"


async def async_execute_function(name: str, args: dict) -> str:
    """Async function executors (for group member queries)."""

    if name == "query_group_member":
        group_id = _current_msg.get("group_id")
        if not group_id:
            return "查群成员需要在群内进行，当前不在群聊中。"

        members = await _fetch_group_members(int(group_id))
        if not members:
            return f"未能获取群 {group_id} 的成员列表。"

        qq = str(args.get("qq_number", "")).strip()
        nick = str(args.get("nickname", "")).strip()

        found = []
        for m in members:
            m_qq = str(m.get("user_id", ""))
            m_nick = m.get("nickname", "")
            m_card = m.get("card", "") or m_nick
            m_role = m.get("role", "member")
            role_label = {"owner": "群主", "admin": "管理员", "member": "成员"}.get(m_role, m_role)

            if qq and m_qq == qq:
                found.append(m)
                break
            if nick and (nick.lower() in m_nick.lower() or nick.lower() in m_card.lower()):
                found.append(m)

        if qq and not nick:
            # Exact QQ lookup
            if found:
                m = found[0]
                return (
                    f"群成员身份：\n"
                    f"  QQ号：{m.get('user_id')}\n"
                    f"  昵称：{m.get('nickname', '未知')}\n"
                    f"  群名片：{m.get('card') or m.get('nickname', '未知')}\n"
                    f"  身份：{role_label}"
                )
            return f"在群中未找到 QQ 号 {qq} 的成员。"
        elif nick:
            if found:
                lines = [f"搜索「{nick}」找到 {len(found)} 个群成员："]
                for m in found[:10]:
                    lines.append(
                        f"  QQ:{m.get('user_id')}  {m.get('card') or m.get('nickname', '?')}  [{role_label}]"
                    )
                return "\n".join(lines)
            return f"在群中未找到包含「{nick}」的成员。"
        else:
            return "请提供 QQ 号或昵称关键字来查群成员。"

    if name == "list_group_members":
        group_id = _current_msg.get("group_id")
        if not group_id:
            return "列出群成员需要在群内进行，当前不在群聊中。"

        members = await _fetch_group_members(int(group_id))
        if not members:
            return f"未能获取群 {group_id} 的成员列表。"

        lines = [f"本群共 {len(members)} 名成员："]
        for m in members[:30]:
            card = m.get("card") or m.get("nickname", "?")
            role_label = {"owner": "群主", "admin": "管理", "member": ""}.get(m.get("role", "member"), "")
            role_str = f" [{role_label}]" if role_label else ""
            lines.append(f"  {m.get('user_id')}  {card}{role_str}")
        if len(members) > 30:
            lines.append(f"  ... 还有 {len(members) - 30} 人未列出")
        return "\n".join(lines)

    return f"未知函数: {name}"


# ─── Message handler ──────────────────────────────────────

async def handle_message(msg: dict):
    global _current_msg
    msg_type = msg.get("message_type", "private")
    raw = msg.get("raw_message", msg.get("message", ""))
    sender_id = msg.get("user_id") or msg.get("sender", {}).get("user_id", "unknown")
    group_id = msg.get("group_id")

    if msg_type == "group" and group_id:
        if not raw.strip().startswith(SPBOT_COMMAND_PREFIX):
            return
        query = raw[len(SPBOT_COMMAND_PREFIX):].strip()
    else:
        query = raw.strip()

    if not query:
        return

    log.info(f"Processing: [{sender_id}] {query[:80]}")

    ck = _chat_key(msg)
    _expire_old(ck)
    _last_active[ck] = time.time()

    # Set global context for function executors
    _current_msg = msg

    # Build messages with context
    context_note = ""
    if msg_type == "group":
        context_note = f"\n[当前在群聊中，群号：{group_id}]"
    else:
        context_note = "\n[当前在私聊中，没有群成员查询功能。涉及群成员的问题请告知用户需要在群里问]"

    history = _history.get(ck, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"[发送者QQ: {sender_id}]{context_note} 用户说：{query}"})

    used_functions = False

    for _ in range(5):
        response = await call_llm(messages, data_mode=used_functions)
        messages.append(response)

        if response.get("tool_calls"):
            used_functions = True
            for tc in response["tool_calls"]:
                func = tc["function"]
                fname = func["name"]
                try:
                    fargs = json.loads(func["arguments"])
                except Exception:
                    fargs = {}

                log.info(f"  → calling {fname}({fargs})")

                # Route async functions separately
                if fname in ("query_group_member", "list_group_members"):
                    result = await async_execute_function(fname, fargs)
                else:
                    result = execute_function(fname, fargs)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        elif response.get("content"):
            reply = response["content"]
            h = _history[ck]
            h.append({"role": "user", "content": query})
            h.append({"role": "assistant", "content": reply})
            _trim_history(ck)
            await send_reply(msg, reply)
            return

    _history[ck].append({"role": "user", "content": query})
    _history[ck].append({"role": "assistant", "content": "抱歉，处理超时，请稍后重试。"})
    _trim_history(ck)
    await send_reply(msg, "抱歉，处理超时，请稍后重试。")


# ─── OneBot sender / receiver ─────────────────────────────

_ws_clients: dict[str, websockets.WebSocketServerProtocol] = {}


async def send_reply(msg: dict, text: str):
    if not _ws_clients:
        log.warning("No NapCatQQ client connected")
        return
    ws = next(iter(_ws_clients.values()))
    msg_type = msg.get("message_type", "private")
    payload = {
        "action": "send_msg",
        "params": {"message_type": msg_type, "message": text},
    }
    if msg_type == "group":
        payload["params"]["group_id"] = msg.get("group_id")
    else:
        payload["params"]["user_id"] = msg.get("user_id") or msg.get("sender", {}).get("user_id")
    try:
        await ws.send(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        log.error(f"Send failed: {e}")


async def handle_ws_connection(ws: websockets.WebSocketServerProtocol):
    peer = ws.remote_address
    log.info(f"NapCatQQ connected from {peer}")

    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Check if this is a response to a pending OneBot API call
            echo = data.get("echo", "")
            if echo and echo in _pending_ob_calls:
                fut = _pending_ob_calls.pop(echo)
                if not fut.done():
                    fut.set_result(data)
                continue

            post_type = data.get("post_type", "")

            if post_type == "meta_event":
                if data.get("meta_event_type") == "lifecycle":
                    self_id = str(data.get("self_id", ""))
                    if self_id:
                        _ws_clients[self_id] = ws
                        log.info(f"Bot self_id: {self_id}")
                continue

            if post_type == "message":
                mt = data.get("message_type", "")
                if mt in ("group", "private"):
                    asyncio.create_task(handle_message(data))

    except websockets.exceptions.ConnectionClosed:
        log.info(f"NapCatQQ disconnected: {peer}")
    except Exception as e:
        log.error(f"WS error: {e}")
    finally:
        for k, v in list(_ws_clients.items()):
            if v is ws:
                del _ws_clients[k]


async def ws_server():
    log.info(f"WebSocket server listening on ws://{SPBOT_LISTEN_HOST}:{SPBOT_LISTEN_PORT}")
    async with serve(handle_ws_connection, SPBOT_LISTEN_HOST, SPBOT_LISTEN_PORT):
        await asyncio.Future()


# ─── Main ─────────────────────────────────────────────────

async def main():
    if not SPBOT_ADMIN_TOKEN:
        log.fatal("SPBOT_ADMIN_TOKEN not set.")
        return
    if not LLM_API_KEY:
        log.fatal("SPBOT_LLM_KEY not set.")
        return
    await ws_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down")
