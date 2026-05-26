"""Entry point for ``python -m pawlia``."""

import argparse
import asyncio
import faulthandler
import logging
import os
import warnings
from typing import Dict, Tuple

# Install signal handlers for SIGSEGV/SIGABRT/SIGBUS/SIGILL so that C-extension
# crashes (aiortc/av/opus/olm) surface a Python traceback to stderr before exit,
# instead of disappearing silently.
faulthandler.enable()


_DARK_GRAY = "\033[90m"
_YELLOW    = "\033[33m"
_RED       = "\033[91m"
_RESET     = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno == logging.DEBUG:
            return f"{_DARK_GRAY}{record.levelname}: {msg}{_RESET}"
        if record.levelno == logging.INFO:
            return f"{_YELLOW}{record.levelname}: {msg}{_RESET}"
        if record.levelno >= logging.ERROR:
            return f"{_RED}{record.levelname}: {msg}{_RESET}"
        return f"{record.levelname}: {msg}"


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_ColorFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    for lib in ("langchain", "langchain_core", "langchain_openai",
                "httpcore", "httpx", "openai", "nio",
                "peewee", "nio.store", "sqlite3"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    # nio logs WARNINGs for schema validation of Matrix events (e.g. empty ICE
    # end-of-candidates, missing user_id in presence); suppress all sub-loggers
    logging.getLogger("nio").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message=".*coroutine.*never awaited"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PawLia - AI Assistant")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--mode", choices=["cli", "server"], default="cli",
        help="cli: interactive terminal | server: all configured interfaces",
    )
    args = parser.parse_args()

    _configure_logging(args.debug)

    if args.debug:
        from pawlia.agents.base import enable_prompt_logging
        enable_prompt_logging()

    asyncio.run(_run(args))


async def _run(args) -> None:
    from pawlia.app import create_app

    app = create_app(config_path=args.config)

    # Start scheduler for proactive reminders/events
    app.scheduler.start()

    # If no models configured, launch web UI for initial setup
    if not app.config.get("models"):
        logging.getLogger("pawlia").info(
            "Keine Modelle konfiguriert — starte Web-UI zur Einrichtung …"
        )
        from pawlia.interfaces.web import start_web
        web_cfg = app.config.get("interfaces", {}).get("web", {})
        try:
            await start_web(app, web_cfg)
        except asyncio.CancelledError:
            pass
        app.scheduler.stop()
        return

    if args.mode == "cli":
        from pawlia.interfaces.cli import start_cli
        await start_cli(app)
        app.scheduler.stop()
        return

    # Server mode: start all interfaces listed under config["interfaces"]
    iface_cfg = app.config.get("interfaces", {})

    import signal
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            pass  # Windows

    log = logging.getLogger("pawlia")

    def _runtime_health() -> Tuple[bool, Dict[str, str]]:
        checks: Dict[str, str] = {}
        ok = True

        sched_task = app.scheduler._task
        if sched_task is None or sched_task.done():
            checks["scheduler"] = "stopped"
            ok = False
        else:
            checks["scheduler"] = "ok"

        for iface_name in sorted(iface_cfg):
            if iface_name == "web":
                continue
            state = interface_health.get(iface_name, "unknown")
            checks[f"interface:{iface_name}"] = state
            if state != "running":
                ok = False

        return ok, checks

    async def _health_watchdog() -> None:
        await asyncio.sleep(90)
        failed_since = None
        while not shutdown.is_set():
            ok, checks = _runtime_health()
            if ok:
                failed_since = None
            else:
                now = asyncio.get_running_loop().time()
                if failed_since is None:
                    failed_since = now
                    log.warning("Runtime health degraded: %s", checks)
                elif now - failed_since >= 120:
                    log.error(
                        "Runtime unhealthy for %.0fs, exiting for container restart: %s",
                        now - failed_since,
                        checks,
                    )
                    os._exit(1)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass

    async def _supervise(name: str, factory) -> None:
        """Run an interface forever, restarting on crash with exponential backoff.

        A single bug in one interface (e.g. an unhandled E2EE error inside
        the matrix sync loop) must not bring down the whole process — the
        container would exit cleanly and Podman's on-failure restart-policy
        would not retry it.
        """
        delay = 1.0
        interface_health[name] = "starting"
        while not shutdown.is_set():
            try:
                interface_health[name] = "running"
                await factory()
                if shutdown.is_set():
                    return
                interface_health[name] = "returned"
                log.warning("Interface '%s' returned without error — restarting", name)
            except asyncio.CancelledError:
                raise
            except Exception:
                interface_health[name] = "crashed"
                log.exception("Interface '%s' crashed — restarting in %.1fs", name, delay)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                pass
            interface_health[name] = "starting"
            delay = min(delay * 2, 60.0)

    tasks = []
    interface_health = {}
    app.interface_health = interface_health

    if "matrix" in iface_cfg:
        from pawlia.interfaces.matrix import start_matrix
        tasks.append(asyncio.create_task(
            _supervise("matrix", lambda: start_matrix(app, iface_cfg["matrix"]))
        ))

    if "discord" in iface_cfg:
        from pawlia.interfaces.discord import start_discord
        tasks.append(asyncio.create_task(
            _supervise("discord", lambda: start_discord(app, iface_cfg["discord"]))
        ))

    if "telegram" in iface_cfg:
        from pawlia.interfaces.telegram import start_telegram
        tasks.append(asyncio.create_task(
            _supervise("telegram", lambda: start_telegram(app, iface_cfg["telegram"]))
        ))

    if "webhook" in iface_cfg:
        from pawlia.interfaces.webhook import start_webhook
        tasks.append(asyncio.create_task(
            _supervise("webhook", lambda: start_webhook(app, iface_cfg["webhook"]))
        ))

    if "openai" in iface_cfg:
        from pawlia.interfaces.openai_compat import start_openai_compat
        tasks.append(asyncio.create_task(
            _supervise("openai", lambda: start_openai_compat(app, iface_cfg["openai"]))
        ))

    from pawlia.interfaces.web import start_web
    tasks.append(asyncio.create_task(
        _supervise("web", lambda: start_web(app, iface_cfg.get("web", {})))
    ))
    tasks.append(asyncio.create_task(_health_watchdog()))

    if not tasks:
        log.error("Server mode: no interfaces configured in config.yaml under 'interfaces'.")
        app.scheduler.stop()
        return

    try:
        await shutdown.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        app.scheduler.stop()


if __name__ == "__main__":
    main()
