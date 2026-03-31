"""CLI interface for PawLia."""

import asyncio
import logging
import signal
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.cli")

_CYAN  = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"

# Shared state for prompt reprinting after notifications
_waiting_for_input = False


def _print_notification(message: str) -> None:
    """Print a notification, overwriting the current prompt line if needed."""
    if _waiting_for_input:
        # Clear the current "You: " prompt line, print notification, reprint prompt
        sys.stdout.write("\r\033[K")  # carriage return + clear line
        sys.stdout.write(f"{_MAGENTA}{message}{_RESET}\n")
        sys.stdout.write("You: ")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\n{_MAGENTA}{message}{_RESET}\n")
        sys.stdout.flush()


async def start_cli(app: "App") -> None:
    """Start an interactive CLI session."""
    global _waiting_for_input

    from pawlia.interfaces.common import (
        build_status, format_status, md_to_text, handle_model_command,
        format_private_toggle, format_bg_enqueue,
    )

    async def _on_interim(text: str) -> None:
        sys.stdout.write(f"{_CYAN}Bot:{_RESET} {text}\n")
        sys.stdout.flush()

    agent = app.make_agent("cli_user", on_interim=_on_interim)
    print("PawLia CLI. Type 'exit' to quit.\n")

    # Register scheduler callback for CLI user
    async def _cli_notify(user_id: str, message: str) -> None:
        if user_id == "cli_user":
            _print_notification(message)

    app.scheduler.register(_cli_notify)

    loop = asyncio.get_running_loop()

    # SIGINT cancels whatever we're currently awaiting
    active_fut = None

    def _on_sigint():
        if active_fut and not active_fut.done():
            active_fut.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler; fall back to signal.signal
        signal.signal(signal.SIGINT, lambda *_: _on_sigint())

    async def _readline() -> str:
        """Read a line from stdin asynchronously so Ctrl+C can cancel it."""
        return await loop.run_in_executor(None, sys.stdin.readline)

    while True:
        sys.stdout.write("You: ")
        sys.stdout.flush()
        _waiting_for_input = True
        active_fut = asyncio.current_task()

        try:
            user_input = await _readline()
            if not user_input:
                print()
                break
            user_input = user_input.rstrip("\n")
        except (EOFError, asyncio.CancelledError):
            print()
            break
        finally:
            _waiting_for_input = False
            active_fut = None

        if user_input.strip().lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue

        app.scheduler.touch_activity("cli_user")

        if user_input.strip().lower() == "/private":
            active = app.memory.toggle_private(agent.session)
            print(f"{md_to_text(format_private_toggle(active))}\n")
            continue

        if user_input.strip().lower().startswith("/thread"):
            message = user_input.strip()[len("/thread"):].strip()
            if not message:
                print("Verwendung: /thread <Nachricht>\n")
                continue
            thread_id = f"cli_{int(time.time())}"
            active_fut = asyncio.current_task()
            try:
                response = await agent.run(message, thread_id=thread_id)
                print(f"{_CYAN}Bot [Thread]:{_RESET} {response}\n")
            except asyncio.CancelledError:
                print("\n(interrupted)")
            except Exception as e:
                logger.error("Error: %s", e)
                print(f"Error: {e}\n")
            continue

        if user_input.strip().lower() == "/status":
            status = build_status(app, "cli_user", agent)
            print(f"\n{md_to_text(format_status(status))}\n")
            continue

        if user_input.strip().lower().startswith("/background"):
            bg_message = user_input.strip()[len("/background"):].strip()
            if not bg_message:
                print("Verwendung: /background <Nachricht>\n")
                continue
            app.scheduler.bg_tasks.enqueue("cli_user", bg_message)
            print(f"{md_to_text(format_bg_enqueue(bg_message))}\n")
            continue

        if user_input.strip().lower().startswith("/model"):
            args_str = user_input.strip()[len("/model"):].strip()
            result = handle_model_command(app, "cli_user", args_str)
            if result.action == "show":
                print(f"Aktives Modell: {result.model}\n")
            else:
                if result.invalidate_agent:
                    agent = app.make_agent("cli_user", on_interim=_on_interim)
                print(f"✓ Modell auf '{result.model}' gesetzt.\n")
            continue

        active_fut = asyncio.current_task()
        try:
            response = await agent.run(user_input)
            print(f"{_CYAN}Bot:{_RESET} {response}\n")
        except asyncio.CancelledError:
            print("\n(interrupted)")
        except Exception as e:
            logger.error("Error: %s", e)
            print(f"Error: {e}\n")

    print("Exiting...")
