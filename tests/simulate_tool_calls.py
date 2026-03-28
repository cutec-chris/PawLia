"""Simulate a long conversation with intermittent tool calls.

Uses the real ChatAgent routing path with a real LLM, but replaces actual
skill execution with deterministic simulated runner outputs. This keeps the
test focused on tool-call accuracy across a longer conversation history while
avoiding external side effects.

Run:
    python -m tests.simulate_tool_calls
    python -m tests.simulate_tool_calls --model qwen3.5:4b
    python -m tests.simulate_tool_calls --list
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


USER_ID = "simulate_tool_calls"


@dataclass
class ConversationTurn:
    name: str
    prompt: str
    expected_skill: Optional[str] = None
    description: str = ""


class SimulatedSkillRunner:
    """Minimal async runner used to avoid real side effects in simulations."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.on_step = None

    async def run(self, query: str) -> str:
        if self.on_step:
            try:
                await self.on_step(f"simulate:{self.skill_name}")
            except Exception:
                pass
        return (
            f"SIMULATED_SKILL_RESULT\n"
            f"skill={self.skill_name}\n"
            f"query={query}\n"
            f"status=ok"
        )


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


def choose_skill(skills: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in skills:
            return name
    return None


def build_turns(skills: Dict[str, Any]) -> List[ConversationTurn]:
    organizer = choose_skill(skills, ["organizer"])
    files = choose_skill(skills, ["files"])
    browser = choose_skill(skills, ["browser"])
    automation = choose_skill(skills, ["automation"])
    search = choose_skill(skills, ["perplexica", "searxng", "researcher"])

    turns: List[ConversationTurn] = [
        ConversationTurn("intro_1", "Hallo, ich bin Chris und arbeite im Vertrieb.", description="Small talk"),
        ConversationTurn("intro_2", "Ich wohne in der Nähe von München.", description="Context setup"),
        ConversationTurn("direct_1", "Was ist die Hauptstadt von Portugal?", description="Direct answer"),
        ConversationTurn("pref_1", "Ich trinke meinen Kaffee schwarz.", description="Context setup"),
    ]

    if organizer:
        turns.append(ConversationTurn(
            "tool_organizer_1",
            "Erinnere mich morgen um 09:00 an die Steuer.",
            expected_skill=organizer,
            description="Reminder dispatch",
        ))

    turns.extend([
        ConversationTurn("direct_2", "Erzähl mir einen kurzen Witz.", description="Direct answer"),
        ConversationTurn("direct_3", "Was ist 17 mal 23?", description="Direct answer"),
    ])

    if search:
        turns.append(ConversationTurn(
            "tool_search_1",
            "Suche nach aktuellen Informationen zu Python 3.14 Type Hints.",
            expected_skill=search,
            description="Search dispatch",
        ))

    turns.extend([
        ConversationTurn("direct_4", "Mein Lieblingsessen ist Sushi.", description="Context setup"),
        ConversationTurn("direct_5", "Wie funktioniert Photosynthese in zwei Sätzen?", description="Direct answer"),
    ])

    if files:
        turns.append(ConversationTurn(
            "tool_files_1",
            "Lege eine Datei notes/test-simulate.txt mit dem Inhalt 'simulated note' an.",
            expected_skill=files,
            description="File operation dispatch",
        ))

    turns.extend([
        ConversationTurn("direct_6", "Ich nutze Arch Linux und Neovim.", description="Context setup"),
        ConversationTurn("direct_7", "Nenne mir drei gute Namen für eine Katze.", description="Direct answer"),
    ])

    if browser:
        turns.append(ConversationTurn(
            "tool_browser_1",
            "Öffne https://example.com und sag mir, was dort steht.",
            expected_skill=browser,
            description="Browser dispatch",
        ))

    turns.extend([
        ConversationTurn("direct_8", "Ich habe zwei Kinder und fahre gerne Rennrad.", description="Context setup"),
        ConversationTurn("direct_9", "Erkläre mir den Unterschied zwischen Python und JavaScript kurz.", description="Direct answer"),
    ])

    if automation:
        turns.append(ConversationTurn(
            "tool_automation_1",
            "Erstelle eine tägliche Automation um 16 Uhr für eine kurze Tageszusammenfassung.",
            expected_skill=automation,
            description="Automation dispatch",
        ))

    turns.extend([
        ConversationTurn("direct_10", "Wie viele Bundesländer hat Deutschland?", description="Direct answer"),
        ConversationTurn("direct_11", "Mein Auto ist ein VW Golf.", description="Context setup"),
        ConversationTurn("direct_12", "Was ist ein gutes Abendessen mit wenig Aufwand?", description="Direct answer"),
    ])

    if organizer:
        turns.append(ConversationTurn(
            "tool_organizer_2",
            "Trag mir am Freitag um 10 Uhr ein Meeting mit Nina ein.",
            expected_skill=organizer,
            description="Late reminder/calendar dispatch",
        ))

    if search:
        turns.append(ConversationTurn(
            "tool_search_2",
            "Suche nach den wichtigsten Neuerungen in Python 3.13 gegenüber 3.12.",
            expected_skill=search,
            description="Late search dispatch",
        ))

    turns.extend([
        ConversationTurn("direct_13", "Ich spreche Deutsch und Englisch.", description="Context setup"),
        ConversationTurn("direct_14", "Was ist Rekursion, ganz kurz erklärt?", description="Direct answer"),
    ])

    return turns


def segment_name(index: int, total: int) -> str:
    third = max(1, total // 3)
    if index < third:
        return "early"
    if index < third * 2:
        return "middle"
    return "late"


async def main() -> None:
    from pawlia.app import create_app

    parser = argparse.ArgumentParser(description="Simulate long conversation tool-call robustness")
    parser.add_argument("--model", help="Override the configured chat model")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--list", action="store_true", help="List the generated conversation turns")
    args = parser.parse_args()

    app = create_app(args.config)
    turns = build_turns(app.skills)

    if args.list:
        safe_print("Planned conversation turns:")
        for i, turn in enumerate(turns, 1):
            expected = turn.expected_skill or "direct"
            safe_print(f"  {i:02d}. [{expected}] {turn.name} - {turn.prompt}")
        return

    user_dir = os.path.join(app.session_dir, USER_ID)
    if os.path.isdir(user_dir):
        shutil.rmtree(user_dir)

    agent = app.make_agent(USER_ID)
    if args.model:
        llm = app.llm.get_with_model(args.model)
        agent.llm = llm
        agent.bound_llm = llm.bind_tools(agent._skill_specs, tool_choice="auto") if agent._skill_specs else llm
        agent.vision_bound_llm = agent.bound_llm

    agent.skill_runner_factory = lambda skill: SimulatedSkillRunner(skill.name)

    safe_print("=" * 72)
    safe_print("Long Conversation Tool-Call Simulation")
    safe_print(f"Model: {args.model or getattr(agent.llm, 'model_name', 'configured default')}")
    safe_print(f"Turns: {len(turns)} | Skills loaded: {', '.join(sorted(agent.skills))}")
    safe_print("=" * 72)

    results: List[Dict[str, Any]] = []
    segment_stats: Dict[str, Dict[str, int]] = {
        "early": {"passed": 0, "total": 0},
        "middle": {"passed": 0, "total": 0},
        "late": {"passed": 0, "total": 0},
    }

    for idx, turn in enumerate(turns, 1):
        seg = segment_name(idx - 1, len(turns))
        t0 = time.time()
        response = await agent.run(turn.prompt)
        elapsed = time.time() - t0

        _, _, tool_calls_info = agent.session.exchanges[-1]
        called_skill = tool_calls_info[0]["name"] if tool_calls_info else None
        called_query = ""
        if tool_calls_info:
            called_query = str(tool_calls_info[0].get("args", {}).get("query", ""))

        if turn.expected_skill:
            success = (
                called_skill == turn.expected_skill
                and bool(called_query.strip())
                and bool(response.strip())
            )
            detail = (
                f"expected={turn.expected_skill} called={called_skill or '-'} "
                f"query_len={len(called_query.strip())} response_len={len(response.strip())}"
            )
        else:
            success = not tool_calls_info and bool(response.strip())
            detail = (
                f"expected=direct called={called_skill or '-'} "
                f"response_len={len(response.strip())}"
            )

        status = "PASS" if success else "FAIL"
        safe_print(
            f"[{idx:02d}/{len(turns)}] [{seg:^6}] [{status}] {turn.name:<20s} "
            f"{elapsed:5.1f}s  {detail}"
        )

        results.append({
            "turn": idx,
            "name": turn.name,
            "segment": seg,
            "success": success,
            "expected_skill": turn.expected_skill,
            "called_skill": called_skill,
            "query_len": len(called_query.strip()),
            "response_len": len(response.strip()),
            "elapsed": elapsed,
        })
        segment_stats[seg]["total"] += 1
        if success:
            segment_stats[seg]["passed"] += 1

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    tool_total = sum(1 for r in results if r["expected_skill"])
    tool_passed = sum(1 for r in results if r["expected_skill"] and r["success"])
    direct_total = total - tool_total
    direct_passed = passed - tool_passed

    safe_print("\n" + "=" * 72)
    safe_print(f"Overall: {passed}/{total} passed ({passed / total * 100:.1f}%)")
    if tool_total:
        safe_print(f"Tool turns: {tool_passed}/{tool_total} passed ({tool_passed / tool_total * 100:.1f}%)")
    if direct_total:
        safe_print(f"Direct turns: {direct_passed}/{direct_total} passed ({direct_passed / direct_total * 100:.1f}%)")
    for name in ("early", "middle", "late"):
        stats = segment_stats[name]
        ratio = (stats["passed"] / stats["total"] * 100.0) if stats["total"] else 0.0
        safe_print(f"{name.title():<6}: {stats['passed']}/{stats['total']} passed ({ratio:.1f}%)")

    failed = [r for r in results if not r["success"]]
    if failed:
        safe_print("\nFailures:")
        for row in failed:
            expected = row["expected_skill"] or "direct"
            safe_print(
                f"  - turn {row['turn']:02d} {row['name']}: expected {expected}, "
                f"got {row['called_skill'] or 'direct'}, query_len={row['query_len']}, response_len={row['response_len']}"
            )
    else:
        safe_print("\nNo failures detected.")


if __name__ == "__main__":
    asyncio.run(main())