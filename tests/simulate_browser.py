"""Simulate the browser skill to test autonomous error recovery.

Runs the SkillRunnerAgent with the browser skill against real websites
and checks whether it can navigate, fill forms, recover from errors,
and complete multi-step tasks autonomously.

Usage:
    python tests/simulate_browser.py                    # run all default tasks
    python tests/simulate_browser.py "custom task"      # run a single custom task
    python tests/simulate_browser.py --list             # show available tasks
    python tests/simulate_browser.py --task 3           # run task 3 only
"""

import asyncio
import logging
import os
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Logging — show SkillRunnerAgent decisions, suppress noise
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)-30s %(levelname)-7s %(message)s",
)
for lib in ("langchain", "langchain_core", "langchain_openai", "httpcore", "httpx", "openai", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("simulate_browser")

SESSION_FILE = os.path.join(os.path.expanduser("~"), ".pawlia_browser.json")

# ---------------------------------------------------------------------------
# Test tasks — each tests a different aspect of autonomous operation
# ---------------------------------------------------------------------------
DEFAULT_TASKS = [
    {
        "name": "simple_fetch",
        "description": "Basic page fetch and data extraction",
        "query": "Go to https://wttr.in/Munich?format=3 and tell me the current weather.",
        "expect_contains": ["Munich"],
    },
    {
        "name": "click_navigation",
        "description": "Open page, click a link, report destination",
        "query": (
            "Open https://example.com, then click the link on the page. "
            "Tell me what the destination page says."
        ),
        "expect_contains": ["IANA", "example"],
    },
    {
        "name": "error_recovery_bad_element",
        "description": "Try a wrong element ID, recover via show",
        "query": (
            "Open https://example.com then click element B99. "
            "If that fails, look at the available elements and click the correct one."
        ),
        "expect_contains": [],  # just needs to not crash
    },
    {
        "name": "form_fill_submit",
        "description": "Find a form, fill it, submit it",
        "query": (
            "Open https://duckduckgo.com, find the search form, "
            "type 'current time in Berlin' into the search input, "
            "and submit the form. Report what the results page shows."
        ),
        "expect_contains": [],
    },
    {
        "name": "back_navigation",
        "description": "Navigate forward then back",
        "query": (
            "Open https://example.com, click the link on the page, "
            "then go back. Confirm you are back on example.com."
        ),
        "expect_contains": ["example.com"],
    },
    {
        "name": "nonexistent_url_recovery",
        "description": "Try a broken URL, then recover with correct one",
        "query": (
            "Open https://thisdomaindoesnotexist12345.com — "
            "if that fails, open https://example.com instead and tell me what it says."
        ),
        "expect_contains": ["Example Domain"],
    },
    {
        "name": "wikipedia_search",
        "description": "Multi-step: open Wikipedia, search, report results",
        "query": (
            "Open https://de.wikipedia.org, find the search form, "
            "search for 'München', and tell me the first paragraph of the article."
        ),
        "expect_contains": ["München"],
    },
    {
        "name": "multi_step_weather",
        "description": "Navigate wttr.in in multiple steps",
        "query": (
            "Open https://wttr.in/Paris and tell me the weather. "
            "Then open https://wttr.in/Tokyo and tell me that weather too. "
            "Compare the two."
        ),
        "expect_contains": ["Paris", "Tokyo"],
    },
]


def clear_browser_session():
    """Delete the browser session file so each test starts fresh."""
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)


async def run_task(app, task: dict, task_num: int, total: int) -> dict:
    """Run a single browser task through the SkillRunnerAgent."""
    from pawlia.agents.skill_runner import SkillRunnerAgent

    skill = app.skills.get("browser")
    if not skill:
        logger.error("Browser skill not found! Available: %s", list(app.skills.keys()))
        return {"success": False, "error": "no browser skill"}

    # Fresh session per task
    clear_browser_session()

    skill_cfg = app.config.get("skill-config", {}).get("browser", {})
    runner = SkillRunnerAgent(
        llm=app.llm.get("skill_runner"),
        skill=skill,
        tool_registry=app.tools,
        context={
            "skill_config": skill_cfg,
            "user_id": "test_browser",
            "session_dir": app.session_dir,
        },
    )

    name = task["name"]
    query = task["query"]
    expect = task.get("expect_contains", [])

    print(f"\n{'='*70}")
    print(f"[{task_num}/{total}] {name}: {task['description']}")
    print(f"{'='*70}")
    print(f"Query: {query}\n")

    t0 = time.time()
    result = await runner.run(query=query)
    elapsed = time.time() - t0

    print(f"\n{'-'*70}")
    print(f"RESULT ({elapsed:.1f}s):")
    print(f"{'-'*70}")
    print(result[:1000] if result else "(empty — agent produced no output)")
    if len(result) > 1000:
        print(f"... ({len(result)} chars total)")
    print()

    # Check expectations
    success = bool(result.strip())
    missing = []
    if expect:
        result_lower = result.lower()
        for word in expect:
            if word.lower() not in result_lower:
                missing.append(word)
        if missing:
            success = False

    status = "PASS" if success else "FAIL"
    detail = f" (missing: {', '.join(missing)})" if missing else ""
    print(f"  >>> [{status}]{detail}")

    return {
        "name": name,
        "success": success,
        "missing": missing,
        "elapsed": elapsed,
        "result_len": len(result),
    }


async def main():
    from pawlia.app import create_app

    app = create_app()

    # Parse args
    if "--list" in sys.argv:
        print("Available tasks:")
        for i, t in enumerate(DEFAULT_TASKS, 1):
            print(f"  {i}. [{t['name']}] {t['description']}")
        return

    if "--task" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--task") + 1]) - 1
        tasks = [DEFAULT_TASKS[idx]]
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        tasks = [{
            "name": "custom",
            "description": "Custom task",
            "query": " ".join(sys.argv[1:]),
            "expect_contains": [],
        }]
    else:
        tasks = DEFAULT_TASKS

    results = []
    for i, task in enumerate(tasks, 1):
        res = await run_task(app, task, i, len(tasks))
        results.append(res)

    # Summary
    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed}/{total} passed")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        detail = f" (missing: {', '.join(r['missing'])})" if r.get("missing") else ""
        print(f"  [{status}] {r['name']:<30s} {r['elapsed']:5.1f}s  {r['result_len']:>5d} chars{detail}")

    if passed < total:
        print(f"\n  {total - passed} task(s) failed.")
    else:
        print(f"\n  All tasks passed!")


if __name__ == "__main__":
    asyncio.run(main())
