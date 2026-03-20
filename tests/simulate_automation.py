"""LLM simulation for the automation system.

Tests whether small models (qwen3.5:4b) correctly dispatch to the organizer
and automation skills. Sends realistic user prompts and checks if the LLM
calls the right skill with reasonable arguments.

Run: python -m tests.simulate_automation [--model qwen3.5:4b]
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen3.5:4b"
API_BASE = "http://192.168.177.120:11434/v1"
API_KEY = "ollama"

# Skill specs exactly as the ChatAgent would present them
SKILL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "organizer",
            "description": (
                "Personal planner for reminders, calendar events, and tasks. "
                "Use when the user wants to: be reminded of something, "
                "plan an event/appointment, or manage personal tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The task or query for this skill",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "automation",
            "description": (
                "Write and schedule automation scripts. Use when the user wants "
                "something to happen automatically or repeatedly (e.g. 'show my "
                "tasks every 5 minutes', 'send me a daily report at 16:00', "
                "'check the weather every hour'). This skill writes the script "
                "and registers the scheduled job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The task or query for this skill",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "files",
            "description": (
                "Read, write, list, and delete files in the user's personal workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The task or query for this skill",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are PawLia, a helpful AI assistant.\n\n"
    "IMPORTANT: You have skills (tools) available. "
    "When a user asks for information that a skill can provide "
    "(routes, train connections, searches, file operations, etc.), "
    "you MUST call the matching skill. NEVER guess or make up answers - "
    "always use the skill to get real data.\n"
    "Only answer directly for simple conversation (greetings, opinions, "
    "general knowledge).\n\n"
    "IMPORTANT: Appointments, events, reminders, and tasks ALWAYS go to "
    "the organizer skill - NEVER to the files skill. "
    "The files skill is ONLY for personal notes and preferences "
    "(name, language, habits), NOT for anything time-related.\n\n"
    "When you learn a persistent fact or preference about the user "
    "(name, language, habits, preferences, etc.), "
    "use the files skill to append it to memory/memory.md."
)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# (prompt, expected_skill_or_None, description)
TEST_CASES: List[Tuple[str, Optional[str], str]] = [
    # Simple reminders -> organizer
    (
        "erinnere mich in 10 minuten an die pizza",
        "organizer",
        "Simple reminder",
    ),
    (
        "remind me in 2 hours to call mom",
        "organizer",
        "English reminder",
    ),
    (
        "sag mir um 16 uhr bescheid dass ich einkaufen muss",
        "organizer",
        "Reminder at specific time",
    ),
    # Events -> organizer
    (
        "ich habe morgen um 14 uhr einen termin in magdeburg",
        "organizer",
        "Calendar event with location",
    ),
    (
        "trag mir am freitag 10 uhr ein meeting ein",
        "organizer",
        "Calendar event",
    ),
    # Tasks -> organizer
    (
        "ich muss bis freitag den bericht fertig haben",
        "organizer",
        "Task with deadline",
    ),
    (
        "neue aufgabe: server backup einrichten, prioritaet hoch",
        "organizer",
        "Task with priority",
    ),
    # Recurring/automated -> automation
    (
        "kannst du mir alle 5 minuten meine aufgaben anzeigen?",
        "automation",
        "Recurring task display",
    ),
    (
        "erstelle mir jeden tag um 16 uhr eine zusammenfassung",
        "automation",
        "Daily automation",
    ),
    (
        "check every hour if there are new files in the downloads folder",
        "automation",
        "Hourly file check automation",
    ),
    (
        "schreib ein script das mir jeden morgen das wetter zeigt",
        "automation",
        "Script for weather automation",
    ),
    # Direct conversation -> None (no skill call)
    (
        "hallo wie gehts?",
        None,
        "Greeting (no skill)",
    ),
    (
        "was ist die hauptstadt von frankreich?",
        None,
        "General knowledge (no skill)",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


async def run_test(llm: Any, prompt: str, expected: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """Send a prompt to the LLM and check if it calls the expected skill.

    Returns (passed, detail_message, actual_skill_called).
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        return False, f"LLM error: {e}", None

    # Check what skill was called
    actual_skill = None
    query = ""
    if response.tool_calls:
        actual_skill = response.tool_calls[0]["name"]
        query = response.tool_calls[0].get("args", {}).get("query", "")

    if expected is None:
        # Expect NO skill call
        if actual_skill is None:
            content = (response.content or "")[:80]
            return True, f"Direct answer: {content}", None
        else:
            return False, f"Called '{actual_skill}' but expected direct answer", actual_skill

    if actual_skill == expected:
        return True, f"-> {actual_skill}(query={query[:60]})", actual_skill
    elif actual_skill is None:
        content = (response.content or "")[:80]
        return False, f"Answered directly instead of calling '{expected}': {content}", None
    else:
        return False, f"Called '{actual_skill}' instead of '{expected}' (query={query[:60]})", actual_skill


async def main(model: str, runs: int = 1):
    safe_print("=" * 70)
    safe_print(f"PawLia Automation Skill Dispatch Simulation")
    safe_print(f"Model: {model} | Runs: {runs}")
    safe_print("=" * 70)

    llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # lower = more deterministic
        base_url=API_BASE,
        api_key=API_KEY,
        timeout=120,
    )
    bound_llm = llm.bind_tools(SKILL_SPECS, tool_choice="auto")

    total = 0
    passed = 0
    failures: List[Tuple[str, str, str]] = []

    for run in range(1, runs + 1):
        if runs > 1:
            safe_print(f"\n--- Run {run}/{runs} ---")

        for prompt, expected, description in TEST_CASES:
            total += 1
            ok, detail, actual = await run_test(bound_llm, prompt, expected)

            status = "[PASS]" if ok else "[FAIL]"
            if ok:
                passed += 1
            else:
                failures.append((description, prompt, detail))

            expected_str = expected or "(direct)"
            safe_print(f"  {status} {description:35s} expect={expected_str:12s} | {detail}")

    # Summary
    safe_print("\n" + "=" * 70)
    safe_print(f"Results: {passed}/{total} passed ({100*passed/total:.0f}%)")

    if failures:
        safe_print(f"\n{len(failures)} failures:")
        for desc, prompt, detail in failures:
            safe_print(f"  - {desc}: \"{prompt}\"")
            safe_print(f"    {detail}")

    safe_print("=" * 70)
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate LLM skill dispatch")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to test (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (for reliability testing)")
    args = parser.parse_args()

    success = asyncio.run(main(args.model, args.runs))
    sys.exit(0 if success else 1)
