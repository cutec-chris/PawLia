"""Web interface for PawLia — single-page app with chat, provider & model management.

A random access token is generated on startup and printed to the console.
Enter it in the browser to authenticate.

Config (under ``interfaces.web``):
    host: 0.0.0.0
    port: 8888
    token: <optional — auto-generated if omitted>
"""

import asyncio
import io
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
import zipfile
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import jinja2
import yaml
from aiohttp import web

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.web")

_PKG_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SKILLS_DIR     = os.path.join(_PKG_ROOT, "skills")
_USER_SKILLS_DIR = os.path.join(_SKILLS_DIR, "user")

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATES_DIR),
    autoescape=False,  # HTML is trusted; JS template literals must not be escaped
)

# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Skill helpers
# ---------------------------------------------------------------------------

def _scan_skills(skill_config: dict) -> list:
    """Scan ALL skills (built-in + user), regardless of config completeness."""
    from pawlia.utils import collect_skill_dirs, parse_frontmatter

    all_dirs = collect_skill_dirs(_SKILLS_DIR)

    result = []
    for skill_path in all_dirs:
        is_user = skill_path.startswith(os.path.abspath(_USER_SKILLS_DIR) + os.sep)
        try:
            fm = parse_frontmatter(os.path.join(skill_path, "SKILL.md"))
            if not fm or not fm.get("name"):
                continue
            name = fm["name"]
            requires = fm.get("metadata", {}).get("requires_config", [])
            current  = skill_config.get(name, {})
            missing  = [k for k in requires if k not in current]
            result.append({
                "name":           name,
                "description":    fm.get("description", ""),
                "version":        str(fm.get("metadata", {}).get("version", "")),
                "author":         fm.get("metadata", {}).get("author", ""),
                "requires_config": requires,
                "config":         current,
                "missing_config": missing,
                "active":         len(missing) == 0,
                "is_user":        is_user,
            })
        except Exception as e:
            logger.debug("Skill scan error at %s: %s", skill_path, e)

    return sorted(result, key=lambda x: x["name"])


def _find_config_path(hint: Optional[str] = None) -> Optional[str]:
    """Locate the active config file on disk."""
    candidates: List[str] = []
    if hint:
        candidates.append(hint)
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for base in (os.getcwd(), pkg_root):
        for name in ("config.yaml", "config.yml", "config.json"):
            candidates.append(os.path.join(base, name))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _write_config(path: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        else:
            json.dump(cfg, f, indent=2, ensure_ascii=False)


async def start_web(app: "App", cfg: Dict) -> None:
    """Start the web interface server."""
    host: str = cfg.get("host", "0.0.0.0")
    port: int = cfg.get("port", 8888)

    # Token: use configured or auto-generate
    token: str = cfg.get("token") or secrets.token_urlsafe(32)

    # Print prominently to console — include token in URL for click-to-login
    border = "=" * 62
    print(f"\n{border}")
    print("  PAWLIA WEB INTERFACE")
    print(f"  URL:   http://localhost:{port}?token={token}")
    print(f"{border}\n")

    config_path = _find_config_path(getattr(app, "config_path", None))
    if not config_path:
        logger.warning("Web: could not locate config file — provider/model edits disabled")

    from pawlia.interfaces.common import AgentCache

    agent_cache = AgentCache(app)
    pending: Dict[str, List[str]] = defaultdict(list)
    sessions: set = set()

    _COOKIE = "pawlia_session"

    def _authed(request: web.Request) -> bool:
        return request.cookies.get(_COOKIE, "") in sessions

    def _unauth() -> web.Response:
        return web.json_response({"error": "unauthorized"}, status=401)

    # ── Static ──────────────────────────────────────────────────────────────

    async def handle_index(request: web.Request) -> web.Response:
        # Auto-login via ?token= query parameter
        url_token = request.query.get("token")
        html = _jinja_env.get_template("index.html").render()
        resp = web.Response(text=html, content_type="text/html")
        if url_token == token:
            sid = secrets.token_urlsafe(32)
            sessions.add(sid)
            resp.set_cookie(_COOKIE, sid, httponly=True, samesite="Strict", max_age=86400 * 7)
        return resp

    # ── Auth ────────────────────────────────────────────────────────────────

    async def handle_auth(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        if body.get("token") != token:
            return web.json_response({"error": "invalid token"}, status=401)
        sid = secrets.token_urlsafe(32)
        sessions.add(sid)
        resp = web.json_response({"ok": True})
        resp.set_cookie(_COOKIE, sid, httponly=True, samesite="Strict", max_age=86400 * 7)
        return resp

    async def handle_logout(request: web.Request) -> web.Response:
        sessions.discard(request.cookies.get(_COOKIE, ""))
        resp = web.json_response({"ok": True})
        resp.del_cookie(_COOKIE)
        return resp

    # ── Chat ────────────────────────────────────────────────────────────────

    async def handle_chat(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        user_id = body.get("user_id", "web_user")
        message = body.get("message", "").strip()
        images = body.get("images") or None
        thread_id = body.get("thread_id") or None

        if not message and not images:
            return web.json_response({"error": "empty message"}, status=400)

        # ── Commands (/status, /model, /private, /thread) ──
        lower = message.lower().strip()

        if lower == "/status":
            from pawlia.interfaces.common import build_status, format_status
            agent = agent_cache.get(user_id)
            status = build_status(app, user_id, agent, thread_id=thread_id)
            return web.json_response({"response": format_status(status)})

        if lower.startswith("/model"):
            from pawlia.interfaces.common import handle_model_command
            args_str = message.strip()[len("/model"):].strip()
            result = handle_model_command(app, user_id, args_str, thread_id=thread_id)
            if result.invalidate_agent:
                agent_cache.invalidate(user_id)
            if result.action == "show":
                return web.json_response({"response": f"**Model ({result.ctx_label}):** `{result.model}`"})
            return web.json_response({"response": f"Model auf `{result.model}` gesetzt ({result.ctx_label})."})

        if lower == "/private":
            session = app.memory.load_session(user_id)
            if thread_id:
                active = app.memory.toggle_private_thread(session, thread_id)
            else:
                active = app.memory.toggle_private(session)
            icon = "\U0001f512" if active else "\U0001f513"
            state = "aktiviert" if active else "deaktiviert"
            return web.json_response({"response": f"{icon} Private Mode {state}"})

        if lower.startswith("/background"):
            bg_message = message.strip()[len("/background"):].strip()
            if not bg_message:
                return web.json_response({"response": "_Verwendung: /background <Nachricht>_"})
            task = app.scheduler.bg_tasks.enqueue(user_id, bg_message)
            return web.json_response({
                "response": f"⏳ Aufgabe in Warteschlange: **{bg_message[:60]}**\nWird im Hintergrund verarbeitet wenn das System idle ist.",
            })

        if lower.startswith("/thread"):
            thread_msg = message.strip()[len("/thread"):].strip()
            if not thread_msg:
                return web.json_response({"response": "_Verwendung: /thread <Nachricht>_"})
            new_thread = f"web_{int(time.time())}"
            agent = agent_cache.get(user_id)
            await app.scheduler.acquire_llm()
            try:
                resp = await agent.run(thread_msg, thread_id=new_thread)
            finally:
                app.scheduler.release_llm()
            return web.json_response({"response": resp, "thread_id": new_thread})

        # ── Normal message (SSE stream) ──
        app.scheduler.touch_activity(user_id)
        logger.info("Web chat: %s: %s", user_id, message[:80])

        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        async def _sse(event: str, data: dict) -> None:
            try:
                payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                await resp.write(payload.encode("utf-8"))
            except (ConnectionResetError, ConnectionError):
                pass

        try:
            agent = agent_cache.get(user_id)

            async def _on_skill_start(skill_name: str, query: str) -> None:
                await _sse("skill_start", {"skill": skill_name, "query": query})

            async def _on_skill_step(step_text: str) -> None:
                await _sse("skill_step", {"text": step_text})

            async def _on_skill_done(skill_name: str) -> None:
                await _sse("skill_done", {"skill": skill_name})

            await app.scheduler.acquire_llm()
            try:
                response = await agent.run(
                    message, images=images, thread_id=thread_id,
                    on_skill_start=_on_skill_start,
                    on_skill_step=_on_skill_step,
                    on_skill_done=_on_skill_done,
                )
            finally:
                app.scheduler.release_llm()
            await _sse("done", {"response": response})
        except Exception as e:
            logger.error("Web chat error: %s", e, exc_info=True)
            await _sse("error", {"error": "internal error"})
        finally:
            await resp.write_eof()

        return resp

    async def handle_notifications(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        uid = request.query.get("user_id", "web_user")
        msgs = pending.pop(uid, [])
        return web.json_response({"notifications": msgs})

    # ── Providers ────────────────────────────────────────────────────────────

    async def handle_get_providers(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"providers": {}})
        try:
            data = _read_config(config_path)
            return web.json_response({"providers": data.get("providers", {})})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_save_providers(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"error": "no config file"}, status=500)
        try:
            body = await request.json()
            data = _read_config(config_path)
            data["providers"] = body.get("providers", {})
            _write_config(config_path, data)
            logger.info("Web: providers updated")
            return web.json_response({"ok": True})
        except Exception as e:
            logger.error("Web: error saving providers: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    # ── Models ───────────────────────────────────────────────────────────────

    async def handle_get_models(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"models": {}})
        try:
            data = _read_config(config_path)
            return web.json_response({"models": data.get("models", {})})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_save_models(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"error": "no config file"}, status=500)
        try:
            body = await request.json()
            data = _read_config(config_path)
            data["models"] = body.get("models", {})
            _write_config(config_path, data)
            logger.info("Web: models updated")
            return web.json_response({"ok": True})
        except Exception as e:
            logger.error("Web: error saving models: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    # ── Skills ───────────────────────────────────────────────────────────────

    async def handle_list_skills(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        try:
            cfg_data = _read_config(config_path) if config_path else {}
            skills = _scan_skills(cfg_data.get("skill-config", {}))
            return web.json_response({"skills": skills})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_skill_upload(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()

        os.makedirs(_USER_SKILLS_DIR, exist_ok=True)

        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            return web.json_response({"error": "no file field in form"}, status=400)

        zip_data = await field.read()
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_data))
        except zipfile.BadZipFile:
            return web.json_response({"error": "ungültige ZIP-Datei"}, status=400)

        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)

            # Find SKILL.md — at root or one level deep
            skill_root = None
            if os.path.isfile(os.path.join(tmpdir, "SKILL.md")):
                skill_root = tmpdir
            else:
                for entry in sorted(os.listdir(tmpdir)):
                    ep = os.path.join(tmpdir, entry)
                    if os.path.isdir(ep) and os.path.isfile(os.path.join(ep, "SKILL.md")):
                        skill_root = ep
                        break

            if not skill_root:
                return web.json_response({"error": "Keine SKILL.md im ZIP gefunden"}, status=400)

            from pawlia.utils import parse_frontmatter
            fm = parse_frontmatter(os.path.join(skill_root, "SKILL.md"))
            if not fm or not fm.get("name"):
                return web.json_response({"error": "SKILL.md hat keinen Namen"}, status=400)

            skill_name = fm["name"]
            dest = os.path.join(_USER_SKILLS_DIR, skill_name)

            # Replace existing
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(skill_root, dest)

        # Install deps + compile workflows
        from pawlia.install_skill_deps import install_skills
        await install_skills(_USER_SKILLS_DIR)

        logger.info("Web: skill '%s' installed", skill_name)
        return web.json_response({
            "ok": True,
            "name": skill_name,
            "message": "Neustart erforderlich, damit der Skill aktiv wird.",
        })

    async def handle_skill_delete(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        skill_name = request.match_info["name"]
        dest = os.path.join(_USER_SKILLS_DIR, skill_name)
        # Path traversal guard
        if not os.path.abspath(dest).startswith(os.path.abspath(_USER_SKILLS_DIR) + os.sep):
            return web.json_response({"error": "ungültiger Name"}, status=400)
        if not os.path.isdir(dest):
            return web.json_response({"error": "Skill nicht gefunden"}, status=404)
        shutil.rmtree(dest)
        logger.info("Web: skill '%s' deleted", skill_name)
        return web.json_response({"ok": True})

    async def handle_get_skill_config(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"skill_config": {}})
        try:
            data = _read_config(config_path)
            return web.json_response({"skill_config": data.get("skill-config", {})})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_save_skill_config(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"error": "no config file"}, status=500)
        try:
            body = await request.json()
            data = _read_config(config_path)
            data["skill-config"] = body.get("skill_config", {})
            _write_config(config_path, data)
            logger.info("Web: skill-config updated")
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ── Settings (interfaces, tts, transcription) ─────────────────────────

    # Sections exposed for editing via the web UI.
    _SETTINGS_SECTIONS = ("interfaces", "tts", "transcription")

    async def handle_get_settings(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"settings": {}})
        try:
            data = _read_config(config_path)
            settings = {k: data.get(k, {}) for k in _SETTINGS_SECTIONS}
            return web.json_response({"settings": settings})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_save_settings(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        if not config_path:
            return web.json_response({"error": "no config file"}, status=500)
        try:
            body = await request.json()
            data = _read_config(config_path)
            for section in _SETTINGS_SECTIONS:
                if section in body.get("settings", {}):
                    data[section] = body["settings"][section]
            _write_config(config_path, data)
            logger.info("Web: settings updated")
            return web.json_response({"ok": True})
        except Exception as e:
            logger.error("Web: error saving settings: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    # ── Memory users / graph ────────────────────────────────────────────────

    async def handle_memory_users(request: web.Request) -> web.Response:
        """Return list of user IDs that have a memory index graph."""
        if not _authed(request):
            return _unauth()
        session_dir = os.path.join(_PKG_ROOT, "session")
        if not os.path.isdir(session_dir):
            return web.json_response({"users": []})
        users = sorted(
            uid for uid in os.listdir(session_dir)
            if os.path.isfile(os.path.join(
                session_dir, uid, "memory_index",
                "graph_chunk_entity_relation.graphml",
            ))
        )
        return web.json_response({"users": users})

    async def handle_memory_graph(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()

        user_id = request.query.get("user_id", "web_user")
        graphml_path = os.path.join(
            _PKG_ROOT, "session", user_id, "memory_index",
            "graph_chunk_entity_relation.graphml",
        )
        if not os.path.isfile(graphml_path):
            return web.json_response({"nodes": [], "edges": []})

        try:
            import networkx as nx

            def _read_graph():
                g = nx.read_graphml(graphml_path)
                nodes = []
                for nid, data in g.nodes(data=True):
                    nodes.append({
                        "id": nid,
                        "label": data.get("entity_name", data.get("label", nid)),
                        "type": data.get("entity_type", data.get("type", "")),
                        "description": data.get("description", ""),
                    })
                edges = []
                for src, tgt, data in g.edges(data=True):
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "label": data.get("description", data.get("label", "")),
                        "weight": float(data.get("weight", 1.0)),
                    })
                return nodes, edges

            nodes, edges = await asyncio.to_thread(_read_graph)
            return web.json_response({"nodes": nodes, "edges": edges})
        except Exception as e:
            logger.error("Memory graph error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    # ── Scheduler callback ───────────────────────────────────────────────────

    async def _notify(user_id: str, message: str) -> None:
        pending[user_id].append(message)

    app.scheduler.register(_notify)

    # ── Setup / bootstrap ──────────────────────────────────────────────────

    async def handle_setup_status(request: web.Request) -> web.Response:
        if not _authed(request):
            return _unauth()
        cfg_data = _read_config(config_path) if config_path else {}
        has_providers = bool(cfg_data.get("providers"))
        has_models = bool(cfg_data.get("models"))
        return web.json_response({
            "has_providers": has_providers,
            "has_models": has_models,
        })

    async def handle_setup_auto(request: web.Request) -> web.Response:
        """Check if ollama is reachable, pull qwen3.5:latest, write config."""
        if not _authed(request):
            return _unauth()

        import aiohttp as _aiohttp

        ollama_base = "http://localhost:11434"
        # 1) Check connectivity
        try:
            async with _aiohttp.ClientSession() as sess:
                async with sess.get(f"{ollama_base}/api/tags", timeout=_aiohttp.ClientTimeout(total=5)) as r:
                    if r.status != 200:
                        return web.json_response({"error": "Ollama nicht erreichbar", "phase": "connect"}, status=502)
        except Exception:
            return web.json_response({"error": "Ollama nicht erreichbar — läuft der Container?", "phase": "connect"}, status=502)

        # 2) Pull model (this can take a while)
        model_name = "qwen3.5:latest"
        try:
            async with _aiohttp.ClientSession() as sess:
                async with sess.post(
                    f"{ollama_base}/api/pull",
                    json={"name": model_name},
                    timeout=_aiohttp.ClientTimeout(total=600),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        return web.json_response({"error": f"Pull fehlgeschlagen: {body}", "phase": "pull"}, status=502)
                    # Consume the streaming response to wait for completion
                    async for _ in r.content:
                        pass
        except Exception as e:
            return web.json_response({"error": f"Pull fehlgeschlagen: {e}", "phase": "pull"}, status=502)

        # 3) Write config
        cfg_path = config_path
        if not cfg_path:
            cfg_path = os.path.join(_PKG_ROOT, "config.yaml")

        cfg_data = _read_config(cfg_path) if os.path.isfile(cfg_path) else {}

        if not cfg_data.get("providers"):
            cfg_data["providers"] = {}
        cfg_data["providers"]["ollama"] = {
            "apiBase": "http://localhost:11434/v1",
            "apiKey": "ollama",
            "timeout": 240,
            "keepAlive": -1,
        }

        if not cfg_data.get("models"):
            cfg_data["models"] = {}
        cfg_data["models"]["fast"] = {
            "model": model_name,
            "provider": "ollama",
            "temperature": 0.7,
        }

        if not cfg_data.get("agents"):
            cfg_data["agents"] = {}
        cfg_data["agents"]["default"] = "fast"

        _write_config(cfg_path, cfg_data)

        # Reload app config so LLMFactory picks up changes on next request
        app.config.update(cfg_data)
        app.llm = __import__("pawlia.llm", fromlist=["LLMFactory"]).LLMFactory(app.config)

        logger.info("Auto-setup: ollama + %s configured", model_name)
        return web.json_response({"ok": True, "model": model_name})

    # ── Build & start ────────────────────────────────────────────────────────

    webapp = web.Application()
    webapp.router.add_get("/",                    handle_index)
    webapp.router.add_post("/api/auth",           handle_auth)
    webapp.router.add_post("/api/logout",         handle_logout)
    webapp.router.add_post("/api/chat",           handle_chat)
    webapp.router.add_get("/api/notifications",   handle_notifications)
    webapp.router.add_get("/api/providers",       handle_get_providers)
    webapp.router.add_post("/api/providers",      handle_save_providers)
    webapp.router.add_get("/api/models",          handle_get_models)
    webapp.router.add_post("/api/models",         handle_save_models)
    webapp.router.add_get("/api/skills",          handle_list_skills)
    webapp.router.add_post("/api/skills/upload",  handle_skill_upload)
    webapp.router.add_delete("/api/skills/{name}", handle_skill_delete)
    webapp.router.add_get("/api/skill-config",    handle_get_skill_config)
    webapp.router.add_post("/api/skill-config",   handle_save_skill_config)
    webapp.router.add_get("/api/settings",        handle_get_settings)
    webapp.router.add_post("/api/settings",       handle_save_settings)
    webapp.router.add_get("/api/setup-status",    handle_setup_status)
    webapp.router.add_post("/api/setup/auto",     handle_setup_auto)
    webapp.router.add_get("/api/memory/users",    handle_memory_users)
    webapp.router.add_get("/api/memory/graph",    handle_memory_graph)

    runner = web.AppRunner(webapp)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Web interface: http://%s:%d", host, port)

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        logger.info("Web interface: stopped")
