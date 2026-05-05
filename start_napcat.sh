#!/bin/bash
# ScreenPlan QQ Bot — NapCatQQ Docker startup script
# Place at /opt/napcat/start.sh
#
# Usage: bash /opt/napcat/start.sh
#
# Once running:
#   1. Open http://<server_ip>:6099/webui in your browser
#   2. Scan the QR code with your phone's QQ app
#   3. Bot will stay online persistently
#   4. spbot will auto-connect via ws://localhost:3001

set -e

CONTAINER_NAME="napcat"
IMAGE="lmq8267/napcat:latest"

# Check if image exists
if ! docker images "$IMAGE" --format '{{.Repository}}' | grep -q napcat; then
    echo "Pulling $IMAGE..."
    docker pull "$IMAGE"
fi

# Stop and remove existing container
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Start NapCatQQ with OneBot v11 WebSocket
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart always \
    -p 3001:3001 \
    -p 6099:6099 \
    -v /opt/napcat/data:/app/data \
    -v /opt/napcat/config:/app/napcat/config \
    -e NAPCAT_UID=$(id -u) \
    -e NAPCAT_GID=$(id -g) \
    "$IMAGE"

echo ""
echo "✅ NapCatQQ container started"
echo ""
echo "Next steps:"
echo "  1. Open WebUI: http://$(curl -s ifconfig.me):6099/webui"
echo "  2. Scan QR code with phone QQ"
echo "  3. After login, start spbot: systemctl start spbot"
echo ""
echo "Check logs: docker logs -f napcat"
echo "Check status: docker ps | grep napcat"
