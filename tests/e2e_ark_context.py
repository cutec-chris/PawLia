#!/usr/bin/env python3
"""E2E test: ARK Ascended multi-turn conversation via OpenAI-compat API.

Tests context retention, memory system, and researcher skill integration
using ARK: Survival Ascended topics.

Prerequisites:
    - PawLia running in server mode with config.openai-eval.yaml
    - OpenAI-compat interface on 127.0.0.1:11445

Usage:
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_context.py [--setup-only] [--test-only]
"""

import argparse
import asyncio
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.request
import urllib.error

API_BASE = "http://127.0.0.1:11445/v1"
USER_ID = "ark_e2e_tester"
RESEARCH_PROJECT = "ark-ascended"

ARK_WIKI_URLS = [
    "https://ark.wiki.gg/wiki/Taming",
    "https://ark.wiki.gg/wiki/Creature_IDs",
    "https://ark.wiki.gg/wiki/Base_Creature_Speeds",
    "https://ark.wiki.gg/wiki/Breeding",
    "https://ark.wiki.gg/wiki/Resources",
    "https://ark.wiki.gg/wiki/The_Island",
    "https://ark.wiki.gg/wiki/Obelisks",
    "https://ark.wiki.gg/wiki/Bosses",
    "https://ark.wiki.gg/wiki/Ark:_Survival_Ascended",
]

CONVERSATION_TURNS = [
    {
        "msg": "Erstelle ein Forschungsprojekt namens 'ark-ascended' zum Thema ARK Survival Ascended. Lade Informationen zum Taming, Breeding und den Bossen herunter.",
        "desc": "Setup researcher project + load data",
        "expect_keywords": ["ark-ascended", "Forschungsprojekt", "projekt", "research"],
        "timeout": 600,
    },
    {
        "msg": "Was weisst du ueber das Taming-System in ARK Ascended?",
        "desc": "Test researcher-grounded answer on taming",
        "expect_keywords": ["taming", "knockout", "passive", "food", "narcotic", "torpor"],
        "timeout": 600,
    },
    {
        "msg": "Ich spiele gerade auf The Island. Welche Bosse gibt es da und welche Dinos sollte ich fuer den Bosskampf tame?",
        "desc": "Test context-aware follow-up (references previous taming topic + new boss topic)",
        "expect_keywords": ["broodmother", "megapithecus", "dragon", "boss", "rex", "therizinosaurus", "dinosaurier", "island"],
        "timeout": 600,
    },
    {
        "msg": "Wie funktioniert Breeding in ARK? Ich moechte meine Rexe fuer den Bosskampf breeden.",
        "desc": "Test cross-topic context: breeding + earlier boss/taming context",
        "expect_keywords": ["breeding", "mating", "egg", "incubation", "mutation", "stat", "rex"],
        "timeout": 600,
    },
    {
        "msg": "Welche Ressourcen brauche ich fuer die Boss-Arena auf The Island?",
        "desc": "Test long-context: combines boss + island + resources knowledge",
        "expect_keywords": ["artifact", "obelix", "obelisk", "tribute", "dungeon", "cave"],
        "timeout": 600,
    },
    {
        "msg": "Erinnere dich: Welche Dinos hast du mir fuer den Bosskampf empfohlen, und warum?",
        "desc": "MEMORY TEST: must reference earlier recommendation from turn 3",
        "expect_keywords": ["rex", "empfohlen", "boss", "theri", "dinos"],
        "timeout": 600,
    },
    {
        "msg": "Zusammenfassung: Was haben wir alles ueber ARK Ascended besprochen?",
        "desc": "MEMORY TEST: must summarize all topics (taming, bosses, breeding, resources, island)",
        "expect_keywords": ["taming", "boss", "breeding", "resource", "island", "rex"],
        "timeout": 600,
    },
    {
        "msg": "Wie schnell ist ein Rex im Vergleich zu einem Spino?",
        "desc": "Test researcher query on specific stat data",
        "expect_keywords": ["speed", "rex", "spino", "geschwindigkeit", "sprint", "bewegung"],
        "timeout": 600,
    },
    {
        "msg": "Was war nochmal das erste Thema ueber das wir gesprochen haben?",
        "desc": "MEMORY TEST: must remember first conversation topic (taming/researcher setup)",
        "expect_keywords": ["taming", "forschungsprojekt", "zahmen", "research", "erst"],
        "timeout": 600,
    },
    {
        "msg": "Ich plane einen Boss-Raid mit 10 Tamed Rexen. Gibt es Tipps fuer die Ausruestung und Sattel?",
        "desc": "Test multi-skill: researcher data + conversation context synthesis",
        "expect_keywords": ["sattel", "saddle", "armor", "rüstung", "rex", "boss", "tip", "weapon", "waffe"],
        "timeout": 600,
    },
]


def _api_request(endpoint: str, data: dict = None, method: str = "GET", timeout: int = 30):
    url = f"{API_BASE}{endpoint}"
    body = json.dumps(data).encode() if data else None
    headers = {
        "Content-Type": "application/json",
        "X-User-Id": USER_ID,
    }
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        return {"error": body_text}, e.code
    except urllib.error.URLError as e:
        return {"error": str(e)}, 0


def _chat(messages: list, timeout: int = 60) -> dict:
    data = {
        "model": "fast",
        "messages": messages,
        "stream": False,
        "user": USER_ID,
    }
    result, status = _api_request("/chat/completions", data, method="POST", timeout=timeout)
    return result, status


def _check_server():
    result, status = _api_request("/models")
    if status != 200:
        print(f"ERROR: PawLia server not reachable at {API_BASE}")
        print(f"  Start with: PYTHONPATH=. .venv/bin/python -m pawlia --config config.openai-eval.yaml --mode server")
        return False
    models = result.get("data", [])
    print(f"Server OK — {len(models)} model(s) available: {[m['id'] for m in models]}")
    return True


def _setup_research():
    print(f"\n{'='*60}")
    print("SETUP: Creating research project and loading ARK data")
    print(f"{'='*60}")

    session_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "session")
    research_dir = os.path.join(session_dir, USER_ID, "workspace", "research", RESEARCH_PROJECT)

    if os.path.exists(research_dir):
        print(f"  Removing existing project: {research_dir}")
        shutil.rmtree(research_dir)

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ["PAWLIA_USER_ID"] = USER_ID
    os.environ["PAWLIA_SESSION_DIR"] = session_dir

    from skills.researcher.scripts.researcher import (
        cmd_create, cmd_add, cmd_list,
    )

    user_dir = os.path.join(session_dir, USER_ID, "workspace", "research")
    os.makedirs(user_dir, exist_ok=True)

    async def _setup():
        print("  Creating project 'ark-ascended'...")
        await cmd_create(
            pathlib.Path(user_dir),
            "ark-ascended",
            "ARK: Survival Ascended — Taming, Breeding, Bosses, Resources, Creatures",
        )

        for i, url in enumerate(ARK_WIKI_URLS):
            print(f"  [{i+1}/{len(ARK_WIKI_URLS)}] Loading: {url}")
            try:
                await cmd_add(
                    pathlib.Path(user_dir),
                    "ark-ascended",
                    url,
                )
            except Exception as e:
                print(f"    WARN: {e}")

        await cmd_list(pathlib.Path(user_dir))

    asyncio.run(_setup())
    print("  Setup complete.\n")


def _run_conversation():
    print(f"\n{'='*60}")
    print(f"E2E CONVERSATION TEST — {len(CONVERSATION_TURNS)} turns")
    print(f"{'='*60}")

    if not _check_server():
        sys.exit(1)

    messages = []
    passed = 0
    failed = 0
    results = []

    for i, turn in enumerate(CONVERSATION_TURNS, 1):
        print(f"\n--- Turn {i}/{len(CONVERSATION_TURNS)}: {turn['desc']} ---")
        print(f"  User: {turn['msg']}")

        messages.append({"role": "user", "content": turn["msg"]})

        t0 = time.time()
        result, status = _chat(messages, timeout=turn["timeout"])
        elapsed = time.time() - t0

        if status == 0:
            print(f"  ERROR: Connection failed ({elapsed:.1f}s)")
            failed += 1
            results.append({"turn": i, "desc": turn["desc"], "status": "CONNECTION_ERROR", "time": elapsed})
            continue

        if status != 200:
            error_data = result.get("error", {}) if isinstance(result, dict) else {}
            error_msg = error_data.get("message", str(result))[:200] if isinstance(error_data, dict) else str(result)[:200]
            print(f"  ERROR: HTTP {status} — {error_msg}")
            failed += 1
            results.append({"turn": i, "desc": turn["desc"], "status": f"HTTP_{status}", "time": elapsed})
            continue

        choices = result.get("choices", [])
        if not choices:
            print(f"  ERROR: No choices in response")
            failed += 1
            results.append({"turn": i, "desc": turn["desc"], "status": "NO_CHOICES", "time": elapsed})
            continue

        assistant_msg = choices[0].get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": assistant_msg})

        preview = assistant_msg[:200].replace("\n", " ")
        print(f"  Bot ({elapsed:.1f}s): {preview}{'...' if len(assistant_msg) > 200 else ''}")

        lower_response = assistant_msg.lower()
        matched = [kw for kw in turn["expect_keywords"] if kw.lower() in lower_response]
        match_ratio = len(matched) / len(turn["expect_keywords"]) if turn["expect_keywords"] else 1.0

        if match_ratio >= 0.2:
            status_str = "PASS"
            passed += 1
        else:
            status_str = "FAIL"
            failed += 1

        print(f"  [{status_str}] Keywords matched: {matched} ({match_ratio:.0%})")

        results.append({
            "turn": i,
            "desc": turn["desc"],
            "status": status_str,
            "keywords_matched": matched,
            "keyword_ratio": match_ratio,
            "response_len": len(assistant_msg),
            "time": elapsed,
        })

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.0f}%)")
    print(f"{'='*60}")

    for r in results:
        icon = "+" if r["status"] == "PASS" else "!"
        print(f"  [{icon}] Turn {r['turn']}: {r['desc']} — {r['status']} ({r.get('time', 0):.1f}s)")

    results_path = os.path.join(os.path.dirname(__file__), "e2e_ark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_path}")

    return failed == 0


def _check_session_state():
    print(f"\n{'='*60}")
    print("SESSION STATE CHECK")
    print(f"{'='*60}")

    session_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "session", USER_ID)

    memory_dir = os.path.join(session_dir, "workspace", "memory")
    today = time.strftime("%Y-%m-%d")
    daily_log = os.path.join(memory_dir, f"{today}.md")

    if os.path.isfile(daily_log):
        with open(daily_log, encoding="utf-8") as f:
            content = f.read()
        exchanges = content.count("[") - content.count("[Version:")
        print(f"  Daily log: {len(content)} chars, ~{max(0, exchanges)} exchanges")
        print(f"  Contains 'taming': {'taming' in content.lower()}")
        print(f"  Contains 'boss': {'boss' in content.lower()}")
        print(f"  Contains 'breeding': {'breeding' in content.lower()}")
    else:
        print(f"  WARNING: No daily log at {daily_log}")

    summary_path = os.path.join(memory_dir, "context_summary.md")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = f.read()
        print(f"  Summary: {len(summary)} chars")
        if summary.strip():
            print(f"    Preview: {summary[:150]}...")
    else:
        print(f"  No summary yet (expected for < 20 exchanges)")

    memory_path = os.path.join(session_dir, "workspace", "memory.md")
    if os.path.isfile(memory_path):
        with open(memory_path, encoding="utf-8") as f:
            mem = f.read()
        print(f"  memory.md: {len(mem)} chars")

    research_dir = os.path.join(session_dir, "workspace", "research", RESEARCH_PROJECT)
    if os.path.isdir(research_dir):
        docs = [f for f in os.listdir(research_dir) if f.endswith(".md") and f != "README.md"]
        print(f"  Research docs: {len(docs)} files")
    else:
        print(f"  WARNING: No research project directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E ARK Ascended Context/Memory Test")
    parser.add_argument("--setup-only", action="store_true", help="Only setup research data")
    parser.add_argument("--test-only", action="store_true", help="Only run conversation test")
    parser.add_argument("--check-state", action="store_true", help="Only check session state")
    args = parser.parse_args()

    if args.check_state:
        _check_session_state()
        sys.exit(0)

    if not args.test_only:
        _setup_research()

    if not args.setup_only:
        ok = _run_conversation()
        _check_session_state()
        sys.exit(0 if ok else 1)
