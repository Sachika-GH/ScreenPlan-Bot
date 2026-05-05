#!/usr/bin/env python3
"""
ScreenPlan QQ Bot — Privacy-aware AI Agent via OneBot v11 WebSocket.

Architecture:
  Group chat  → general AI + group member lookup (NO ScreenPlan access)
  Private DM  → login with email/password → query OWN ScreenPlan data only

Dependencies: websockets, httpx
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

# ─── Config ───────────────────────────────────────────────

SPBOT_LISTEN_HOST = os.environ.get("SPBOT_LISTEN_HOST", "0.0.0.0")
SPBOT_LISTEN_PORT = int(os.environ.get("SPBOT_LISTEN_PORT", "3001"))
SPBOT_COMMAND_PREFIX = os.environ.get("SPBOT_PREFIX", "/")
SCREENPLAN_HOST = os.environ.get("SPBOT_SCREENPLAN_HOST", "http://localhost:5051")
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


# ─── User authentication ──────────────────────────────────

_user_sessions: dict[str, dict] = {}  # {qq_number_str: {token, user_id, display_name, email}}


def _get_user_token(sender_id: str) -> str | None:
    """Get stored JWT for a QQ user, removing expired sessions."""
    sid = str(sender_id)
    s = _user_sessions.get(sid)
    if s:
        # JWT expires after 30 days — we trust the backend to reject if expired
        return s["token"]
    return None


def _api(path: str, token: str | None = None, method: str = "GET", body: dict | None = None) -> dict:
    """Call ScreenPlan API (public or authenticated)."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{SCREENPLAN_HOST}/api/{path}"
    try:
        if method == "POST":
            r = httpx.post(url, headers=headers, json=body, timeout=30)
        else:
            r = httpx.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        detail = ""
        try:
            detail = ": " + e.response.json().get("error", "")
        except Exception:
            pass
        return {"error": f"请求失败 ({e.response.status_code if hasattr(e, 'response') else '?'}){detail}"}
    except Exception as e:
        return {"error": str(e)}


# ─── OneBot API helper ────────────────────────────────────

_ob_req_id = 0
_pending_ob_calls: dict[str, asyncio.Future] = {}


async def onebot_api(action: str, params: dict) -> dict:
    global _ob_req_id
    if not _ws_clients:
        return {"error": "No NapCatQQ connection"}
    ws = next(iter(_ws_clients.values()))
    _ob_req_id += 1
    req_id = _ob_req_id
    payload = json.dumps({"action": action, "params": params, "echo": str(req_id)}, ensure_ascii=False)
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _pending_ob_calls[str(req_id)] = future
    try:
        await ws.send(payload)
        return await asyncio.wait_for(future, timeout=15)
    except asyncio.TimeoutError:
        _pending_ob_calls.pop(str(req_id), None)
        return {"error": "OneBot API timeout"}
    except Exception as e:
        _pending_ob_calls.pop(str(req_id), None)
        return {"error": str(e)}


_group_member_cache: dict[str, tuple[float, list[dict]]] = {}
GROUP_CACHE_TTL = 300


async def _fetch_group_members(group_id: int) -> list[dict]:
    gk = str(group_id)
    now = time.time()
    if gk in _group_member_cache:
        ts, members = _group_member_cache[gk]
        if now - ts < GROUP_CACHE_TTL:
            return members
    resp = await onebot_api("get_group_member_list", {"group_id": group_id})
    if "error" in resp or resp.get("status") != "ok":
        return []
    members = resp.get("data", [])
    _group_member_cache[gk] = (now, members)
    return members


# ─── LLM caller ───────────────────────────────────────────

async def call_llm(messages: list, tools: list, data_mode: bool = False) -> dict:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "tools": tools,
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


# ─── Function definitions ─────────────────────────────────

# 🔒 Private-chat-only: ScreenPlan functions (require login)
_PRIVATE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "login_screenplan",
            "description": "登录 ScreenPlan 账号。用户必须提供邮箱和密码。仅在私聊中可用。登录后用户才能查看自己的屏幕使用数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "ScreenPlan 注册邮箱"},
                    "password": {"type": "string", "description": "ScreenPlan 密码"},
                },
                "required": ["email", "password"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_my_usage",
            "description": "查询当前登录用户自己的屏幕使用摘要（今日或指定日期）。只有在用户已登录后才能调用。不需要提供 user_id，会自动获取当前用户的数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "日期 YYYY-MM-DD，默认今天"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_my_timeline",
            "description": "查询当前登录用户自己的时间线详情。只有在用户已登录后才能调用。不需要提供 user_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "日期 YYYY-MM-DD，默认今天"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_help",
            "description": "显示当前环境下可用的功能列表。当用户问「你能做什么」「帮助」「help」时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 👥 Group-chat-only: general AI + group member lookup
_GROUP_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_group_member",
            "description": "在 QQ 群中按 QQ 号或昵称/群名片查找群成员身份。",
            "parameters": {
                "type": "object",
                "properties": {
                    "qq_number": {"type": "string", "description": "要查找的 QQ 号（纯数字）"},
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
            "description": "列出当前 QQ 群的所有成员。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_help",
            "description": "显示当前群聊中可用的功能列表。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Prompt for GROUP context — NO ScreenPlan
SYSTEM_PROMPT_GROUP = """你是一个友好的 QQ 群 AI 助手。

你的能力：
① 通用 AI 知识 —— 编程、数学、常识、翻译、闲聊
② QQ 群管理 —— 查群成员身份、列成员列表（query_group_member / list_group_members）
③ 帮助 —— show_help

重要规则：
- 你无法查询任何 ScreenPlan 屏幕时间数据。群聊中不提供此功能。
- 关于屏幕时间的问题，回复：「请私聊我并先登录 ScreenPlan 账号来查看你的使用数据」
- 绝不编造数据。不知道就说不知道。
- QQ 不支持 Markdown。纯文本分行回复。保持简洁。"""

# Prompt for PRIVATE context — requires login for ScreenPlan
SYSTEM_PROMPT_PRIVATE = """你是一个个人 AI 助手，通过私聊提供服务。

你的能力：
① 通用 AI 知识 —— 编程、数学、常识、翻译、闲聊
② ScreenPlan 个人数据 —— 登录后查看自己的屏幕使用情况
③ 帮助 —— show_help

隐私规则（非常重要）：
- 用户必须先通过 login_screenplan 登录（提供邮箱+密码），才能查看屏幕数据
- 用户只能查看自己的数据。query_my_usage 和 query_my_timeline 自动使用当前登录用户的数据
- 你无法查询其他用户的屏幕数据。这是隐私保护设计
- 如果用户发来消息「登录 xxx@qq.com xxxxxx」，主动调用 login_screenplan 帮助登录
- 如果用户未登录就问屏幕数据，回复：「请先登录。发送你的 ScreenPlan 邮箱和密码，格式：登录 your@email.com 你的密码」
- 登录成功后告知用户已登录，然后用户可以问「我的屏幕时间」「我今天的timeline」等
- 绝不要把任何 ScreenPlan 数据发到群里
- QQ 不支持 Markdown。纯文本分行，时长用「小时 分钟」如 5h33min"""

_current_msg: dict = {}


# ─── Function executors ───────────────────────────────────

def execute_function(name: str, args: dict) -> str:
    """Synchronous function executors."""
    sender_id = str(_current_msg.get("user_id") or _current_msg.get("sender", {}).get("user_id", "0"))

    if name == "login_screenplan":
        email = args.get("email", "").strip()
        password = args.get("password", "").strip()
        if not email or not password:
            return "请同时提供邮箱和密码。格式：登录 your@email.com 你的密码"

        data = _api("auth/login", method="POST", body={"email": email, "password": password})
        if "error" in data:
            return f"登录失败：{data['error']}"

        token = data.get("access_token", "")
        if not token:
            return "登录失败，未收到有效凭证。"

        _user_sessions[sender_id] = {
            "token": token,
            "user_id": data.get("user_id", 0),
            "display_name": data.get("display_name", "?"),
            "email": email,
        }
        log.info(f"User {sender_id} logged in as {data.get('display_name')} (ID:{data.get('user_id')})")
        return (
            f"登录成功！你好，{data.get('display_name', '?')}。\n"
            f"你可以问我：我的屏幕时间 / 我今天的timeline"
        )

    if name == "query_my_usage":
        token = _get_user_token(sender_id)
        if not token:
            return "你尚未登录。请先发送：登录 your@email.com 你的密码"

        dt = (args.get("date") or "").strip() or date.today().isoformat()
        data = _api("usage/summary?date=" + dt, token=token)
        if "error" in data:
            return f"查询失败：{data['error']}"

        total = data.get("total_minutes_all_devices", 0)
        devs = data.get("devices", [])
        name = _user_sessions.get(sender_id, {}).get("display_name", "你")
        lines = [f"{name}（{dt}）屏幕使用："]
        lines.append(f"  总屏幕时间：{total:.0f}min（{total/60:.1f}h）")
        if data.get("overlap_minutes", 0) > 0:
            lines.append(f"  多设备重叠：{data['overlap_minutes']:.0f}min")
        for d in devs:
            lines.append(
                f"  {d['device_name']}（{d['platform']}）：{d['total_minutes']:.0f}min  "
                f"学习{d['learning_pct']:.0f}% 娱乐{d['entertainment_pct']:.0f}%"
            )
        if not devs:
            lines.append("  当日暂无设备活动数据")
        return "\n".join(lines)

    if name == "query_my_timeline":
        token = _get_user_token(sender_id)
        if not token:
            return "你尚未登录。请先发送：登录 your@email.com 你的密码"

        dt = (args.get("date") or "").strip() or date.today().isoformat()
        data = _api("usage/timeline/full?date=" + dt, token=token)
        if "error" in data:
            return f"查询失败：{data['error']}"

        devs = data.get("devices", [])
        name = _user_sessions.get(sender_id, {}).get("display_name", "你")
        total_events = sum(d["event_count"] for d in devs)
        lines = [f"{name}（{dt}）时间线：共 {total_events} 个事件"]
        for d in devs:
            lines.append(f"  {d['device_name']}（{d['platform']}）：{d['event_count']} 事件")
            for ev in d.get("events", [])[-5:]:
                ts = ev["timestamp"].split("T")[1][:5] if "T" in ev["timestamp"] else ev["timestamp"]
                lines.append(f"    {ts}  {ev['app_name']} [{ev['category']}]")
        if not devs:
            lines.append("  当日暂无活动数据")
        return "\n".join(lines)

    if name == "show_help":
        msg_type = _current_msg.get("message_type", "private")
        if msg_type == "group":
            return (
                "群聊中我能帮你：\n"
                "  💬 通用问答 —— 编程、数学、翻译、闲聊\n"
                "  👥 查群成员 —— \"群里xxx是谁\"\n"
                "  📋 列群成员 —— \"列出群成员\"\n\n"
                "私聊中可用 ScreenPlan 屏幕时间查询（需先登录）。\n"
                "私聊发送「登录」查看登录指引。"
            )
        logged_in = _get_user_token(sender_id)
        if logged_in:
            display = _user_sessions.get(sender_id, {}).get("display_name", "你")
            return (
                f"当前已登录：{display}\n\n"
                "ScreenPlan：\n"
                "  • 我的屏幕时间 / 我今天的timeline\n\n"
                "通用：\n"
                "  • 编程 / 数学 / 翻译 / 闲聊\n\n"
                "发送「退出登录」可注销。"
            )
        return (
            "在私聊中我能帮你：\n\n"
            "📊 ScreenPlan 屏幕管理\n"
            "  1. 先登录：发送「登录 your@email.com 你的密码」\n"
            "  2. 查数据：问我「我的屏幕时间」「我今天的timeline」\n\n"
            "💬 通用 AI\n"
            "  • 编程 / 数学 / 翻译 / 闲聊\n\n"
            "⚠️ 群聊中不提供 ScreenPlan 数据查询。"
        )

    return f"未知函数: {name}"


async def async_execute_function(name: str, args: dict) -> str:
    """Async function executors (group member queries)."""
    if name == "query_group_member":
        group_id = _current_msg.get("group_id")
        if not group_id:
            return "查群成员需要在群内进行。"

        members = await _fetch_group_members(int(group_id))
        if not members:
            return "未能获取群成员列表。"

        qq = str(args.get("qq_number", "")).strip()
        nick = str(args.get("nickname", "")).strip()
        found = []
        for m in members:
            m_qq = str(m.get("user_id", ""))
            m_nick = m.get("nickname", "")
            m_card = m.get("card", "") or m_nick
            role_label = {"owner": "群主", "admin": "管理", "member": "成员"}.get(m.get("role", "member"), "")
            if qq and m_qq == qq:
                found.append((m, role_label))
                break
            if nick and (nick.lower() in m_nick.lower() or nick.lower() in m_card.lower()):
                found.append((m, role_label))

        if qq and not nick:
            if found:
                m, rl = found[0]
                return (
                    f"  QQ号：{m.get('user_id')}\n"
                    f"  昵称：{m.get('nickname', '?')}\n"
                    f"  群名片：{m.get('card') or m.get('nickname', '?')}\n"
                    f"  身份：{rl}"
                )
            return f"群中未找到 QQ 号 {qq} 的成员。"
        elif nick:
            if found:
                lines = [f"搜索「{nick}」找到 {len(found)} 人："]
                for m, rl in found[:10]:
                    lines.append(f"  QQ:{m.get('user_id')}  {m.get('card') or m.get('nickname','?')}  [{rl}]")
                return "\n".join(lines)
            return f"未找到含「{nick}」的成员。"
        return "请提供 QQ 号或昵称。"

    if name == "list_group_members":
        group_id = _current_msg.get("group_id")
        if not group_id:
            return "列出群成员需要在群内进行。"
        members = await _fetch_group_members(int(group_id))
        if not members:
            return "未能获取群成员列表。"
        lines = [f"本群共 {len(members)} 人："]
        for m in members[:30]:
            card = m.get("card") or m.get("nickname", "?")
            rl = {"owner": "群主", "admin": "管理", "member": ""}.get(m.get("role", ""), "")
            lines.append(f"  {m.get('user_id')}  {card}" + (f" [{rl}]" if rl else ""))
        if len(members) > 30:
            lines.append(f"  ... 还有 {len(members) - 30} 人")
        return "\n".join(lines)

    return f"未知函数: {name}"


# ─── Message handler ──────────────────────────────────────

async def handle_message(msg: dict):
    global _current_msg
    msg_type = msg.get("message_type", "private")
    raw = msg.get("raw_message", msg.get("message", ""))
    sender_id = str(msg.get("user_id") or msg.get("sender", {}).get("user_id", "0"))
    group_id = msg.get("group_id")

    if msg_type == "group" and group_id:
        if not raw.strip().startswith(SPBOT_COMMAND_PREFIX):
            return
        query = raw[len(SPBOT_COMMAND_PREFIX):].strip()
    else:
        query = raw.strip()

    if not query:
        return

    # Handle explicit logout
    if msg_type == "private" and query.strip().lower() in ("退出登录", "logout", "注销"):
        uid = str(_current_msg.get("user_id") or msg.get("sender", {}).get("user_id", "0"))
        if uid in _user_sessions:
            name = _user_sessions.pop(uid, {}).get("display_name", "?")
            await send_reply(msg, f"已退出 {name} 的登录。")
        else:
            await send_reply(msg, "你当前未登录。")
        return

    log.info(f"Processing: [{sender_id}] {query[:80]}")

    ck = _chat_key(msg)
    _expire_old(ck)
    _last_active[ck] = time.time()
    _current_msg = msg

    # Select context: group vs private
    if msg_type == "group":
        tools = _GROUP_FUNCTIONS
        system_prompt = SYSTEM_PROMPT_GROUP
        context_note = f"\n[群聊，群号：{group_id}]"
    else:
        tools = _PRIVATE_FUNCTIONS
        system_prompt = SYSTEM_PROMPT_PRIVATE
        session = _user_sessions.get(sender_id)
        if session:
            context_note = f"\n[私聊] 当前已登录 ScreenPlan：{session.get('display_name')} (ID:{session.get('user_id')})"
        else:
            context_note = "\n[私聊] 用户尚未登录 ScreenPlan。如需查屏幕数据，先让用户登录。"

    history = _history.get(ck, [])
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"[发送者QQ: {sender_id}]{context_note} 用户说：{query}"})

    used_functions = False

    for _ in range(5):
        response = await call_llm(messages, tools, data_mode=used_functions)
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

    await send_reply(msg, "抱歉，处理超时，请稍后重试。")


# ─── OneBot sender / receiver ─────────────────────────────

_ws_clients: dict[str, websockets.WebSocketServerProtocol] = {}


async def send_reply(msg: dict, text: str):
    if not _ws_clients:
        return
    ws = next(iter(_ws_clients.values()))
    mt = msg.get("message_type", "private")
    payload = {"action": "send_msg", "params": {"message_type": mt, "message": text}}
    if mt == "group":
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
    if not LLM_API_KEY:
        log.fatal("SPBOT_LLM_KEY not set.")
        return
    await ws_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down")
