"""Entry point for ``python -m pawlia``."""

import argparse
import asyncio
import logging
import warnings


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
                "httpcore", "httpx", "openai", "nio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    # nio logs WARNINGs for schema validation of Matrix events (e.g. empty ICE
    # end-of-candidates, missing user_id in presence); suppress all sub-loggers
    logging.getLogger("nio").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message=".*coroutine.*never awaited"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PawLia - AI Assistant")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--mode", choices=["cli", "server"], default="cli",
        help="cli: interactive terminal | server: all configured interfaces",
    )
    args = parser.parse_args()

    _configure_logging(args.debug)

    asyncio.run(_run(args))


async def _run(args) -> None:
    from pawlia.app import create_app

    app = create_app(config_path=args.config)

    # Start scheduler for proactive reminders/events
    app.scheduler.start()

    if args.mode == "cli":
        from pawlia.interfaces.cli import start_cli
        await start_cli(app)
        app.scheduler.stop()
        return

    # Server mode: start all interfaces listed under config["interfaces"]
    iface_cfg = app.config.get("interfaces", {})
    tasks = []

    if "matrix" in iface_cfg:
        from pawlia.interfaces.matrix import start_matrix
        tasks.append(asyncio.create_task(
            start_matrix(app, iface_cfg["matrix"])
        ))

    if "telegram" in iface_cfg:
        from pawlia.interfaces.telegram import start_telegram
        tasks.append(asyncio.create_task(
            start_telegram(app, iface_cfg["telegram"])
        ))

    if "webhook" in iface_cfg:
        from pawlia.interfaces.webhook import start_webhook
        tasks.append(asyncio.create_task(
            start_webhook(app, iface_cfg["webhook"])
        ))

    from pawlia.interfaces.web import start_web
    tasks.append(asyncio.create_task(
        start_web(app, iface_cfg.get("web", {}))
    ))

    if not tasks:
        logging.getLogger("pawlia").error(
            "Server mode: no interfaces configured in config.json under 'interfaces'."
        )
        app.scheduler.stop()
        return

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    main()
