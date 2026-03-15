"""Simulate a longer conversation to test summarization."""

import asyncio
import logging
import shutil
import os

# Colored logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
for lib in ("langchain", "langchain_core", "langchain_openai", "httpcore", "httpx", "openai"):
    logging.getLogger(lib).setLevel(logging.WARNING)

USER_ID = "test_summary"

# Varied conversation to test what the summary keeps/discards
MESSAGES = [
    "hi, ich bin Chris",
    "Zeitzone ist gmt+1",
    "du bist PawLia",
    "ich wohne bei München",
    "wie wird das wetter morgen?",
    "was ist die hauptstadt von frankreich?",
    "ich fahre gerne rennrad und arbeite als softwareentwickler",
    "kannst du mir ein rezept für pasta carbonara geben?",
    "ich mag keine pilze, merk dir das bitte",
    "was ist der unterschied zwischen python und javascript?",
    "mein lieblingsessen ist sushi",
    "wie spät ist es gerade?",
    "ich nutze arch linux btw",
    "erzähl mir einen witz",
    "ich trinke meinen kaffee schwarz",
    "was ist ein guter name für eine katze?",
    "ich hab zwei kinder",
    "wie funktioniert photosynthese?",
    "ich spreche deutsch und englisch",
    "was ist dein lieblingsfilm?",
    "mein auto ist ein VW Golf",
    "kannst du mir bei einem python projekt helfen?",
    "ich benutze neovim als editor",
]


async def main():
    from pawlia.app import create_app

    # Clean previous test session
    session_dir = os.path.join(os.path.dirname(__file__), "session", USER_ID)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)

    app = create_app()
    agent = app.make_agent(USER_ID)

    print(f"\n{'='*60}")
    print(f"Starting conversation simulation ({len(MESSAGES)} messages)")
    print(f"Summary triggers at {agent.session.exchange_count} exchanges")
    print(f"{'='*60}\n")

    for i, msg in enumerate(MESSAGES, 1):
        print(f"\n--- Message {i}/{len(MESSAGES)} ---")
        print(f"User: {msg}")
        response = await agent.run(msg)
        print(f"Bot:  {response[:150]}{'...' if len(response) > 150 else ''}")

        # Check if summary was generated
        if agent.session.summary:
            print(f"\n>>> SUMMARY (after msg {i}):\n{agent.session.summary}")

        # Small delay to not hammer the API
        await asyncio.sleep(0.5)

    # Final state
    print(f"\n{'='*60}")
    print("FINAL STATE")
    print(f"{'='*60}")
    print(f"Exchange count: {agent.session.exchange_count}")
    print(f"\nFinal summary:\n{agent.session.summary or '(none)'}")

    # Read summary from disk
    summary_path = os.path.join(session_dir, "workspace", "memory", "context_summary.md")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            print(f"\nSummary on disk:\n{f.read()}")


if __name__ == "__main__":
    asyncio.run(main())
