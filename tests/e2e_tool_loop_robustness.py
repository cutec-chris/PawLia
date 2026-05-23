#!/usr/bin/env python3
"""E2E test: multi-turn tool-call robustness (GLM 1214 fix verification).

Tests the context summarization, orphan-repair, and tool-result compression
paths across a real ARK-themed conversation.  The system decides which skills
to use — no explicit skill names in the prompts.

Uses gpt-oss-120b in high-thinking mode for the chat agent.

Run:
    PYTHONPATH=. .venv/bin/python tests/e2e_tool_loop_robustness.py
"""

import asyncio
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

USER_ID = "e2e_robustness_tester"

ARK_TURNS = [
    {
        "msg": (
            "Lade Informationen zu ARK Survival Ascended herunter: "
            "https://ark.wiki.gg/wiki/Taming und "
            "https://ark.wiki.gg/wiki/Breeding. "
            "Speicher sie in einem Ordner ark-notes."
        ),
        "desc": "Fetch ARK wiki data (multi-tool: researcher + files)",
    },
    {
        "msg": (
            "Was weisst du ueber das Taming-System in ARK Ascended? "
            "Such in den gerade runtergeladenen Notizen."
        ),
        "desc": "Researcher-grounded answer on taming",
    },
    {
        "msg": (
            "Ich spiele auf The Island und will einen Bosskampf machen. "
            "Welche Dinos brauche ich und wie breed ich die?"
        ),
        "desc": "Cross-context: breeding + boss + taming",
    },
    {
        "msg": (
            "Zusammenfassung: Was haben wir alles ueber ARK besprochen? "
            "Liste die Themen auf."
        ),
        "desc": "Memory: summarize all ARK topics",
    },
]

EXPECT_KEYWORDS = [
    ["taming", "breeding", "information", "daten", "ark"],
    ["taming", "zahmen", "knockout", "narcotic", "torpor"],
    ["breeding", "zucht", "rex", "boss", "mating"],
    ["taming", "boss", "breeding", "rex", "island"],
]


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


async def run_turn(agent, turn: dict, turn_idx: int) -> Dict[str, Any]:
    msg = turn["msg"]
    t0 = time.time()
    try:
        response = await agent.run(msg)
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "idx": turn_idx,
            "desc": turn["desc"],
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "time": elapsed,
            "response": "",
            "tool_turns": 0,
            "keywords_matched": [],
            "keyword_ratio": 0,
            "response_len": 0,
            "response_preview": "",
        }

    elapsed = time.time() - t0

    # Count tool turns in this response
    exchanges = agent.session.exchanges[-1] if agent.session and agent.session.exchanges else []
    tc_info = exchanges[2] if len(exchanges) >= 3 else None
    tool_count = len(tc_info) if tc_info else 0

    # Check keywords
    lower = response.lower()
    expected = EXPECT_KEYWORDS[turn_idx]
    matched = [kw for kw in expected if kw.lower() in lower]
    match_ratio = len(matched) / len(expected) if expected else 1.0

    if match_ratio >= 0.15:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "idx": turn_idx,
        "desc": turn["desc"],
        "status": status,
        "tool_turns": tool_count,
        "keywords_matched": matched,
        "keyword_ratio": match_ratio,
        "response_len": len(response.strip()),
        "time": elapsed,
        "response_preview": response[:200].replace("\n", " "),
    }


async def main() -> None:
    from pawlia.app import create_app

    app = create_app(None)

    # Blow away old state for a clean run
    user_dir = os.path.join(app.session_dir, USER_ID)
    if os.path.isdir(user_dir):
        import shutil
        shutil.rmtree(user_dir)

    # Seed completed identity files so the agent skips bootstrap
    os.makedirs(os.path.join(user_dir, "workspace"), exist_ok=True)
    _SEED_FILES = {
        "identity.md": (
            "# identity.md\n\n"
            "- **Name:** ARKbot\n"
            "- **Creature:** AI Companion\n"
            "- **Vibe:** helpful, direct\n"
            "- **Emoji:** 🦖\n"
            "- **Avatar:** \n"
        ),
        "soul.md": (
            "# SOUL.md\n\n"
            "- **Core Truth:** Be helpful and factual\n"
            "- **Boundaries:** No harmful content\n"
            "- **Vibe:** Practical ARK survival expert\n"
        ),
        "user.md": (
            "# user.md\n\n"
            "**Name:** ARK Survivor\n"
            "**Timezone:** Europe/Berlin\n"
            "**Language:** German\n"
        ),
    }
    for filename, content in _SEED_FILES.items():
        with open(os.path.join(user_dir, "workspace", filename), "w", encoding="utf-8") as f:
            f.write(content)

    # Use GLM-5.1 (Z.AI) — the model that had the 1214 error.
    # Inject a model config entry and point the chat agent to it.
    models_cfg = app.config.get("models") or {}
    models_cfg["glm51"] = {
        "model": "GLM-5.1",
        "provider": "zai",
        "temperature": 0.7,
    }
    agents_cfg = app.config.get("agents") or {}
    agents_cfg["chat"] = "glm51"
    agents_cfg["default"] = "glm51"
    models_cfg["glm51"] = models_cfg["glm51"]
    app.config["models"] = models_cfg
    app.config["agents"] = agents_cfg

    agent = app.make_agent(USER_ID)

    # Silence verbose LLM logging in test output
    import logging
    logging.getLogger("pawlia.agents.chat").setLevel(logging.WARNING)
    logging.getLogger("pawlia.agents.skill_runner").setLevel(logging.WARNING)
    logging.getLogger("pawlia.llm").setLevel(logging.WARNING)

    safe_print("=" * 72)
    safe_print("E2E Tool-Loop Robustness Test (GLM 1214 Fix Verification)")
    safe_print("Model: GLM-5.1 (Z.AI)")
    safe_print(f"Turns: {len(ARK_TURNS)}")
    safe_print(f"Skills: {', '.join(sorted(agent.skills))}")
    safe_print("=" * 72)

    results = []
    for i, turn in enumerate(ARK_TURNS):
        safe_print(f"\n--- Turn {i+1}/{len(ARK_TURNS)}: {turn['desc']} ---")
        safe_print(f"  User: {turn['msg'][:120]}...")

        result = await run_turn(agent, turn, i)
        results.append(result)

        icon = {"PASS": "+", "FAIL": "!", "ERROR": "X"}.get(result["status"], "?")
        safe_print(
            f"  [{icon} {result['status']}] {result['time']:.1f}s "
            f"tools={result.get('tool_turns', 0)} "
            f"kw={result['keyword_ratio']:.0%} "
            f"({len(result['response_preview'])} chars)"
        )
        safe_print(f"  Preview: {result['response_preview']}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ("FAIL", "ERROR"))
    total = len(results)

    safe_print(f"\n{'='*72}")
    safe_print(f"RESULTS: {passed}/{total} passed")
    for r in results:
        icon = {"PASS": "[+]", "FAIL": "[!]", "ERROR": "[X]"}.get(r["status"], "[?]")
        safe_print(
            f"  {icon} Turn {r['idx']:>2}: {r['desc'][:55]:55s} "
            f"{r['status']:10s} tools={r.get('tool_turns',0)} "
            f"kw={r['keyword_ratio']:.0%} {r['time']:.1f}s"
        )

    import json
    results_path = os.path.join(os.path.dirname(__file__), "e2e_robustness_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    safe_print(f"\nResults saved to {results_path}")

    if failed == 0:
        safe_print("\nAll turns passed — GLM 1214 fix verified.")
    else:
        safe_print(f"\n{failed} failure(s) — check the log for details.")
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
