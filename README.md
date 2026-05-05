# ScreenPlan QQ Bot

AI Agent QQ bot with DeepSeek function calling, powered by OneBot v11 protocol.

## Architecture

```
QQ → NapCatQQ ← WebSocket → spbot (this bot) → DeepSeek LLM
                                                  → ScreenPlan Admin API
                                                  → OneBot Group API
```

## Quick Deploy (Ubuntu 22.04+)

```bash
# 1. Install Python deps
pip install websockets httpx

# 2. Edit systemd service with your credentials
cp spbot.service /etc/systemd/system/spbot.service
vim /etc/systemd/system/spbot.service  # set SPBOT_LLM_KEY, SPBOT_ADMIN_TOKEN

# 3. Start
systemctl daemon-reload
systemctl enable --now spbot
```

## Capabilities

| Domain | Functions |
|--------|-----------|
| ScreenPlan | query_user_list, query_user_usage, query_user_timeline, query_server_status |
| QQ Group | query_group_member, list_group_members |
| Help | show_help |
| General AI | Knowledge, coding, math, translation, chat |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| SPBOT_LLM_KEY | Yes | DeepSeek API key |
| SPBOT_ADMIN_TOKEN | Yes | ScreenPlan admin bearer token |
| SPBOT_LISTEN_PORT | No | WebSocket listen port (default 3001) |
| SPBOT_SCREENPLAN_HOST | No | ScreenPlan URL (default http://localhost:5051) |
| SPBOT_LLM_MODEL | No | LLM model (default deepseek-chat) |
