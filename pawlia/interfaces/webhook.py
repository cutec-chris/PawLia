"""Webhook interface for PawLia using aiohttp.

Exposes a simple HTTP POST endpoint that accepts a JSON body
and returns the assistant's response.

Config (in config.yaml under "interfaces.webhook"):
    {
      "host": "0.0.0.0",
      "port": 8080,
      "token": "optional-secret-token"
    }

Request format:
    POST /chat
    Authorization: Bearer <token>   (if token is configured)
    Content-Type: application/json

    {"user_id": "some-user", "message": "hello", "images": ["data:image/png;base64,..."]}

Response:
    {"response": "Hi! How can I help?"}

Notifications (proactive reminders/events):
    GET /notifications?user_id=some-user
    Returns and clears pending notifications for the user.
"""

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

from aiohttp import web

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.webhook")


async def start_webhook(app: "App", cfg: Dict) -> None:
    """Start the webhook HTTP server.

    ``cfg`` is the ``interfaces.webhook`` section of config.yaml.
    """
    host: str = cfg.get("host", "0.0.0.0")
    port: int = cfg.get("port", 8080)
    token: str = cfg.get("token", "")

    from pawlia.interfaces.common import AgentCache, preview_text

    agent_cache = AgentCache(app)
    # Buffer for proactive notifications (polled via GET /notifications)
    pending_notifications: Dict[str, List[str]] = defaultdict(list)

    def _check_auth(request: web.Request) -> bool:
        if not token:
            return True
        return request.headers.get("Authorization", "") == f"Bearer {token}"

    async def handle_chat(request: web.Request) -> web.Response:
        if not _check_auth(request):
            return web.json_response({"error": "unauthorized"}, status=401)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        user_id = body.get("user_id", "webhook_user")
        message = body.get("message", "").strip()
        images = body.get("images") or None  # list of base64 data-URIs
        if not message and not images:
            return web.json_response({"error": "empty message"}, status=400)

        logger.info("Webhook: message from %s: %s (images=%d)", user_id, message[:80], len(images or []))

        try:
            agent = agent_cache.get(user_id)

            async def _on_interim(text: str) -> None:
                pending_notifications[user_id].append(text)

            agent.on_interim = _on_interim
            response = await agent.run(message, images=images)
            logger.info("Webhook: response to %s: %s", user_id, preview_text(response))
            return web.json_response({"response": response})
        except Exception as e:
            logger.error("Webhook: error processing message: %s", e)
            return web.json_response({"error": "internal error"}, status=500)

    async def handle_notifications(request: web.Request) -> web.Response:
        if not _check_auth(request):
            return web.json_response({"error": "unauthorized"}, status=401)

        user_id = request.query.get("user_id", "webhook_user")
        messages = pending_notifications.pop(user_id, [])
        return web.json_response({"notifications": messages})

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    # Register scheduler callback — buffer notifications for polling
    async def _webhook_notify(user_id: str, message: str) -> None:
        pending_notifications[user_id].append(message)

    app.scheduler.register(_webhook_notify)

    webapp = web.Application()
    webapp.router.add_post("/chat", handle_chat)
    webapp.router.add_get("/notifications", handle_notifications)
    webapp.router.add_get("/health", handle_health)

    runner = web.AppRunner(webapp)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Webhook: listening on http://%s:%d", host, port)

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        logger.info("Webhook: stopped")
