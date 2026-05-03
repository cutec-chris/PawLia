"""OpenAI / Ollama-compatible API interface for PawLia.

Allows external tools (Continue.dev, OpenWebUI, Cursor, …) to connect to
picoclaw as if it were an OpenAI-compatible or Ollama endpoint.

Endpoints:
  GET  /v1/models              — list configured models (OpenAI format)
  POST /v1/chat/completions    — chat completions (streaming + non-streaming)
  GET  /api/tags               — model list (Ollama format)
  POST /api/chat               — chat completions (Ollama format)
  GET  /api/version            — version string (Ollama compatibility)

Config (under interfaces.openai):
    host: 0.0.0.0
    port: 11435
    api_key: <optional — if set, require "Authorization: Bearer <key>">
"""

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from aiohttp import web

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.openai_compat")

_FAKE_OLLAMA_VERSION = "0.6.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_list(app: "App") -> List[Dict[str, Any]]:
    """Return a list of model dicts for the configured pawlia models."""
    models = app.config.get("models") or {}
    now = int(time.time())
    result = []
    for key, cfg in models.items():
        raw_model = cfg.get("model", key)
        result.append({
            "id": key,
            "object": "model",
            "created": now,
            "owned_by": "pawlia",
            "model": raw_model,
        })
    return result


def _extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    """Return the content of the last user-role message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content arrays (OpenAI vision format)
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return " ".join(parts)
            return str(content)
    return ""


def _openai_chunk(cid: str, model: str, delta_content: str, finish: Optional[str] = None) -> bytes:
    chunk: Dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": delta_content} if not finish else {},
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()


def _ollama_chunk(model: str, content: str, done: bool = False) -> bytes:
    chunk: Dict[str, Any] = {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        chunk["done_reason"] = "stop"
    return (json.dumps(chunk, ensure_ascii=False) + "\n").encode()


# ---------------------------------------------------------------------------
# Auth middleware factory
# ---------------------------------------------------------------------------

def _make_auth_checker(api_key: Optional[str]):
    def _check(request: web.Request) -> bool:
        if not api_key:
            return True
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer "):].strip() == api_key
        return False

    return _check


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

async def start_openai_compat(app: "App", cfg: Dict[str, Any]) -> None:
    """Start the OpenAI / Ollama-compatible API server."""
    host: str = cfg.get("host", "0.0.0.0")
    port: int = cfg.get("port", 11435)
    api_key: Optional[str] = cfg.get("api_key") or cfg.get("apiKey") or None
    _authed = _make_auth_checker(api_key)

    def _unauth() -> web.Response:
        return web.json_response({"error": {"message": "Unauthorized", "type": "auth_error"}}, status=401)

    # ── /v1/models ────────────────────────────────────────────────────────

    async def handle_list_models(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        return web.json_response({"object": "list", "data": _model_list(app)})

    # ── /v1/chat/completions ──────────────────────────────────────────────

    async def handle_chat_completions(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "invalid JSON"}}, status=400)

        messages: List[Dict[str, Any]] = body.get("messages") or []
        stream: bool = bool(body.get("stream", False))
        model_id: str = str(body.get("model") or "")
        user_id: str = model_id or str(body.get("user") or "openai_api_user")

        user_input = _extract_last_user_message(messages)
        if not user_input:
            return web.json_response({"error": {"message": "no user message found"}}, status=400)

        cid = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        effective_model = model_id or "pawlia"

        agent = app.make_agent(user_id)

        if stream:
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
            await resp.prepare(request)

            # Role delta first
            role_chunk: Dict[str, Any] = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": effective_model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await resp.write(f"data: {json.dumps(role_chunk)}\n\n".encode())

            async def _on_sentence(sentence: str) -> None:
                await resp.write(_openai_chunk(cid, effective_model, sentence + " "))

            try:
                await agent.run_streamed(user_input, on_sentence=_on_sentence)
            except Exception as exc:
                logger.error("openai_compat stream error: %s", exc, exc_info=True)
                err_chunk: Dict[str, Any] = {
                    "id": cid, "object": "chat.completion.chunk",
                    "model": effective_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                await resp.write(f"data: {json.dumps(err_chunk)}\n\n".encode())
            else:
                await resp.write(_openai_chunk(cid, effective_model, "", finish="stop"))

            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp

        # Non-streaming
        try:
            result = await agent.run(user_input)
        except Exception as exc:
            logger.error("openai_compat error: %s", exc, exc_info=True)
            return web.json_response({"error": {"message": str(exc)}}, status=500)

        return web.json_response({
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": effective_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # ── /api/version (Ollama) ─────────────────────────────────────────────

    async def handle_ollama_version(request: web.Request) -> web.Response:
        return web.json_response({"version": _FAKE_OLLAMA_VERSION})

    # ── /api/tags (Ollama) ────────────────────────────────────────────────

    async def handle_ollama_tags(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        models = app.config.get("models") or {}
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entries = []
        for key, cfg in models.items():
            raw_model = cfg.get("model", key)
            entries.append({
                "name": key,
                "model": raw_model,
                "modified_at": now_str,
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "pawlia",
                    "families": ["pawlia"],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                },
            })
        return web.json_response({"models": entries})

    # ── /api/chat (Ollama) ────────────────────────────────────────────────

    async def handle_ollama_chat(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        messages: List[Dict[str, Any]] = body.get("messages") or []
        stream: bool = body.get("stream", True)
        model_id: str = str(body.get("model") or "")
        user_id = model_id or "ollama_api_user"

        user_input = _extract_last_user_message(messages)
        if not user_input:
            return web.json_response({"error": "no user message found"}, status=400)

        effective_model = model_id or "pawlia"
        agent = app.make_agent(user_id)

        if stream:
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "application/x-ndjson",
                    "Cache-Control": "no-cache",
                },
            )
            await resp.prepare(request)

            async def _on_sentence_ollama(sentence: str) -> None:
                await resp.write(_ollama_chunk(effective_model, sentence + " "))

            try:
                await agent.run_streamed(user_input, on_sentence=_on_sentence_ollama)
            except Exception as exc:
                logger.error("ollama_compat stream error: %s", exc, exc_info=True)

            await resp.write(_ollama_chunk(effective_model, "", done=True))
            await resp.write_eof()
            return resp

        # Non-streaming
        try:
            result = await agent.run(user_input)
        except Exception as exc:
            logger.error("ollama_compat error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

        return web.json_response({
            "model": effective_model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": result},
            "done": True,
            "done_reason": "stop",
        })

    # ── Wire up & start ───────────────────────────────────────────────────

    webapp = web.Application()
    webapp.router.add_get("/v1/models",              handle_list_models)
    webapp.router.add_post("/v1/chat/completions",   handle_chat_completions)
    webapp.router.add_get("/api/version",            handle_ollama_version)
    webapp.router.add_get("/api/tags",               handle_ollama_tags)
    webapp.router.add_post("/api/chat",              handle_ollama_chat)

    runner = web.AppRunner(webapp)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    key_hint = " (API key required)" if api_key else " (no auth)"
    logger.info("OpenAI-compat interface: http://%s:%d%s", host, port, key_hint)
    print(f"\nOpenAI-compatible endpoint: http://localhost:{port}/v1{key_hint}")
    print(f"Ollama-compatible endpoint:  http://localhost:{port}/api\n")

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        logger.info("OpenAI-compat interface: stopped")
