"""CLI interface for PawLia."""

import asyncio
import signal
import sys
import logging
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

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    while True:
        # Read from stdin via event loop (non-blocking)
        fut: asyncio.Future[str] = loop.create_future()
        active_fut = fut

        sys.stdout.write("You: ")
        sys.stdout.flush()
        _waiting_for_input = True

        def _on_readable():
            line = sys.stdin.readline()
            if not fut.done():
                if not line:
                    fut.set_exception(EOFError())
                else:
                    fut.set_result(line.rstrip("\n"))
            loop.remove_reader(sys.stdin)

        loop.add_reader(sys.stdin, _on_readable)

        try:
            user_input = await fut
        except (EOFError, asyncio.CancelledError):
            print()
            break
        finally:
            _waiting_for_input = False

        if user_input.strip().lower() in ("exit", "quit"):
            break
        if not user_input.strip():
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
