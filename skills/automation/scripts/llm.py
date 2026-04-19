#!/usr/bin/env python3
"""Automation LLM harness — robuster LLM-Call für Automations-Skripte.

Stellt einen CLI-Wrapper um die pawlia-LLM-Factory bereit, den Automations-
Skripte als Subprocess aufrufen können (analog zu perplexica/scripts/search.py).

Garantie: Wenn das LLM leer oder zu kurz antwortet, wird mit einer Nudge-
Nachricht wiederholt bis ein Ergebnis vorliegt — oder mit Non-Zero-Exit
abgebrochen (kein stilles Leerergebnis).

Usage (CLI):

    python llm.py --prompt "Fasse zusammen: ..." \\
                  [--system "Du bist..."] \\
                  [--model smart] \\
                  [--retries 4] \\
                  [--min-chars 1]

    # Prompt via stdin:
    echo "Fasse zusammen: ..." | python llm.py --stdin
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys

# Make the pawlia package importable when this script is run directly
# (e.g. `python /app/skills/automation/scripts/llm.py ...`).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.abspath(os.path.join(_HERE, "..", "..", "..")),  # repo root
    "/app",  # container layout
):
    if os.path.isdir(os.path.join(_candidate, "pawlia")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pawlia.config import load_config
from pawlia.llm import LLMFactory


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


NUDGE = (
    "Deine letzte Antwort war leer. "
    "Antworte jetzt direkt mit dem gewünschten Text — keine Erklärung, keine leeren Zeilen."
)


async def call(prompt: str, system: str, model: str | None,
               retries: int, min_chars: int) -> str:
    config = load_config()
    factory = LLMFactory(config)
    llm = factory.get_with_model(model) if model else factory.get("chat")

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    last_raw = ""
    for attempt in range(1, retries + 1):
        response = await llm.ainvoke(messages)
        raw = response.content or ""
        content = raw.strip()
        if content and len(content) >= min_chars:
            return content
        last_raw = raw
        print(
            f"llm.py: attempt {attempt}/{retries} returned "
            f"{len(content)} chars (<{min_chars}), nudging",
            file=sys.stderr,
        )
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=NUDGE))

    raise RuntimeError(
        f"LLM lieferte nach {retries} Versuchen kein verwertbares Ergebnis "
        f"(letzte Roh-Antwort: {last_raw!r})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robuster LLM-Call mit Retry/Nudge für Automations-Skripte.",
    )
    parser.add_argument("--prompt", help="User-Prompt. Alternativ --stdin.")
    parser.add_argument("--stdin", action="store_true",
                        help="Prompt von stdin lesen statt --prompt.")
    parser.add_argument("--system", default="",
                        help="Optionaler System-Prompt.")
    parser.add_argument("--model", default=None,
                        help="Modellname aus config.yaml models: (z.B. 'fast'). "
                             "Default: agents.chat.")
    parser.add_argument("--retries", type=int, default=4,
                        help="Maximale Versuche mit Nudge (Default 4).")
    parser.add_argument("--min-chars", type=int, default=1,
                        help="Mindestlänge des stripped Ergebnisses (Default 1).")
    args = parser.parse_args()

    if args.stdin:
        prompt = sys.stdin.read()
    else:
        prompt = args.prompt or ""
    if not prompt.strip():
        print("llm.py: leerer Prompt", file=sys.stderr)
        sys.exit(2)

    try:
        result = asyncio.run(call(
            prompt=prompt,
            system=args.system,
            model=args.model,
            retries=args.retries,
            min_chars=args.min_chars,
        ))
    except Exception as exc:
        print(f"llm.py: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.stdout.write(result)
    if not result.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
