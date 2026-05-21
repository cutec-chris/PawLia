"""E2E Memory System Comparison — main vs. memory-system-optimization.

Compares three observable behaviors that differ between the branches:

  1. Workspace Search  — BM25 search over workspace .md files, injected into
                         user messages.  Only exists on memory-system-optimization.
  2. Summarization threshold — main uses a fixed exchange-count (≥20 → maybe,
                         ≥30 → force).  The new branch uses token-count against
                         the model's context window.  With gpt-oss (200k tokens)
                         25 short exchanges (~3 k tokens) never trigger a summary.
  3. Topic shift       — WorkspaceSearch.is_topic_shift() detects semantic jumps;
                         present on the new branch, absent on main.

Run:
    python -m tests.e2e_memory_comparison

Model: gpt-oss (must be configured in config.yaml).
Each test creates its own isolated temp session; cleaned up afterward.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time
from textwrap import dedent
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Terminal colours
# ──────────────────────────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RESET = "\033[0m"

_PASS = 0
_FAIL = 0
_SKIP = 0
_results: List[Tuple[str, str, str]] = []   # (status, name, detail)


def _p(*args):
    try:
        print(*args)
    except UnicodeEncodeError:
        print(" ".join(str(a) for a in args).encode("ascii", errors="replace").decode())


def check(name: str, condition: bool, detail: str = "", *, skip: bool = False):
    global _PASS, _FAIL, _SKIP
    if skip:
        _SKIP += 1
        _results.append(("SKIP", name, detail))
        _p(f"  {_DIM}[SKIP]{_RESET} {name}" + (f" — {detail}" if detail else ""))
    elif condition:
        _PASS += 1
        _results.append(("PASS", name, ""))
        _p(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _FAIL += 1
        _results.append(("FAIL", name, detail))
        _p(f"  {_RED}[FAIL]{_RESET} {name}" + (f"\n        {_DIM}{detail}{_RESET}" if detail else ""))


def section(title: str):
    _p(f"\n{_BOLD}{'─'*65}{_RESET}")
    _p(f"{_BOLD}  {title}{_RESET}")
    _p(f"{_BOLD}{'─'*65}{_RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Branch detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_branch_features() -> dict:
    features = {}

    # Workspace search
    try:
        from pawlia.workspace_search import WorkspaceSearch  # noqa: F401
        features["workspace_search"] = True
    except ImportError:
        features["workspace_search"] = False

    # Token-based summarization signature
    import inspect
    from pawlia.memory import MemoryManager
    sig = inspect.signature(MemoryManager.should_summarize)
    features["token_based_summary"] = "summary_threshold_tokens" in sig.parameters

    # Agent LLM resolver (chat.py supports _agent_llm_resolver)
    from pawlia.agents.chat import ChatAgent
    features["agent_llm_resolver"] = hasattr(ChatAgent, "_agent_llm_resolver") or \
        "_agent_llm_resolver" in ChatAgent.__init__.__code__.co_varnames

    return features


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Workspace Search
# ──────────────────────────────────────────────────────────────────────────────

async def test_workspace_search(app, tmp_session_dir: str, features: dict):
    section("Test 1 — Workspace Search")

    if not features["workspace_search"]:
        check(
            "WorkspaceSearch module present",
            False,
            "pawlia.workspace_search not importable — this is main / pre-feature branch",
        )
        check(
            "Model answers from workspace",
            False,
            skip=True,
            detail="workspace search not available",
        )
        check(
            "Topic shift triggers re-search",
            False,
            skip=True,
            detail="workspace search not available",
        )
        return

    check("WorkspaceSearch module present", True)

    # Create workspace with a unique project file.
    # Pre-populate identity files so bootstrap.md is never placed in the workspace
    # (bootstrap detection blocks workspace search — we want to test the search, not bootstrap).
    user_id = "e2e_ws_test"
    workspace = os.path.join(tmp_session_dir, user_id, "workspace")
    os.makedirs(workspace, exist_ok=True)

    for fname, content in [
        ("identity.md", "# PawLia\nDu bist PawLia, ein hilfreicher Assistent.\n"),
        ("soul.md", "## Charakter\nFreundlich, präzise, hilfsbereit.\n"),
        ("user.md", "## Nutzer\nTestnutzer für e2e-Tests.\n"),
    ]:
        with open(os.path.join(workspace, fname), "w", encoding="utf-8") as f:
            f.write(content)

    # ARK: Survival Ascended — fictional game-specific content the model cannot
    # reasonably answer from training data (specific base coordinates, dino XP, etc.)
    #
    # BM25 IDF depends on corpus size. With only one content file the corpus is
    # too small: rare terms get IDF ≈ 0.5, making raw scores fall below the
    # stricter min_raw_score=2.0 threshold. Adding unrelated dummy files brings
    # the corpus to ~20 sections, pushing IDFs for ark-specific terms to ~1.8+
    # and raw scores comfortably above 2.0. This mirrors a realistic workspace.
    for fname, content in [
        ("rezepte.md", dedent("""\
            # Rezepte

            ## Pasta Carbonara
            Zutaten: Spaghetti, Guanciale, Pecorino, Eier, Pfeffer.
            Guanciale anbraten, Pasta kochen, Käse-Ei-Mix einrühren.

            ## Bananenbrot
            Zutaten: Mehl, Zucker, Bananen, Butter, Backpulver.
            Bananen zerdrücken, alles vermischen, bei 180°C backen.
        """)),
        ("buchnotizen.md", dedent("""\
            # Buchnotizen

            ## Der Prozess — Kafka
            Josef K. wird verhaftet ohne bekanntes Vergehen.
            Thema: Bürokratie, Schuld, Absurdität.

            ## Faust — Goethe
            Faust schließt einen Pakt mit Mephistopheles.
            Thema: Wissensdrang, Erlösung, Schuld.
        """)),
        ("reisenotizen.md", dedent("""\
            # Reisenotizen

            ## Portugal 2023
            Lissabon: Alfama-Viertel, Pastéis de Belém, Tram 28.
            Porto: Ribeira-Ufer, Portweinkeller Vila Nova de Gaia.

            ## Japan 2024
            Tokio: Shinjuku, Akihabara, Tsukiji-Markt.
            Kyoto: Arashiyama-Bambuswald, Fushimi Inari.
        """)),
    ]:
        with open(os.path.join(workspace, fname), "w", encoding="utf-8") as f:
            f.write(content)

    project_file = os.path.join(workspace, "ark_meine_basis.md")
    with open(project_file, "w", encoding="utf-8") as f:
        f.write(dedent("""\
            # ARK Survival Ascended — Basis-Notizen

            ## ARK Hauptbasis
            Hauptbasis-Koordinaten: 47.3 / 22.1 (Insel, nahe Crystal Caves)
            - Stromversorgung: 8x Windgeneratoren + 2x Gasgeneratoren
            - Tek-Generator PIN: 7714
            - Kühlschrank-PIN: 3389

            ## ARK Gezähmte Dinos
            Rex "Knabberzahn": Level 312, 42.800 HP, Saddle 124 Armor
            - Argy "Silberwind": Level 287, Geschwindigkeit 267%
            - Quetzal "Mammut": Level 231, Gewicht 5.400

            ## ARK Ressourcen-Spots
            - Obsidian: 61.2 / 14.8 (Bergspitze)
            - Element-Vein: 38.9 / 71.4 (Untere Lava-Zone)
            - Schwarze Perle: Ozean-Trench bei 22.0 / 59.0
        """))

    agent = app.make_agent(user_id)

    # First, verify BM25 search fires and populates session.workspace_refs
    from pawlia.workspace_search import WorkspaceSearch
    ws_instance = WorkspaceSearch(workspace)
    hits = ws_instance.search("ARK Basis Koordinaten")
    check(
        "BM25-Suche findet ark_meine_basis.md",
        any("ark" in (h.path or "").lower() or "ark" in (h.snippet or "").lower() for h in hits),
        f"Hits: {[h.path for h in hits]}",
    )

    # Ask the model — it should call files:read via the injected workspace ref
    t0 = time.monotonic()
    answer = await agent.run("Welche Koordinaten hat meine ARK-Hauptbasis?")
    elapsed = time.monotonic() - t0
    _p(f"  {_DIM}[{elapsed:.1f}s]{_RESET} Antwort: {answer[:300]}")

    answer_lower = answer.lower()
    session_obj = agent.session if hasattr(agent, "session") else None
    ws_refs = getattr(session_obj, "workspace_refs", None) if session_obj else None
    check(
        "session.workspace_refs nach ARK-Frage befüllt",
        ws_refs is not None and len(ws_refs) > 0,
        f"workspace_refs={ws_refs!r}",
    )
    check(
        "Koordinaten 47.3 / 22.1 in Antwort",
        "47" in answer and "22" in answer,
        f"Antwort: {answer[:300]}",
    )

    # Topic shift: ask unrelated, then back to ARK base
    _ = await agent.run("Was ist der Unterschied zwischen Python und Rust?")   # unrelated
    answer2 = await agent.run("Wie heißt mein Rex in ARK und welches Level hat er?")
    _p(f"  {_DIM}Nach Topic-Shift:{_RESET} {answer2[:200]}")
    check(
        "Nach Topic-Shift: Dino-Name aus Workspace (Knabberzahn oder Level 312)",
        "knabberzahn" in answer2.lower() or "312" in answer2,
        f"Antwort: {answer2[:300]}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Summarization threshold
# ──────────────────────────────────────────────────────────────────────────────

async def test_summarization_threshold(app, tmp_session_dir: str, features: dict):
    section("Test 2 — Summarization Threshold")

    from pawlia.memory import MemoryManager, MAX_EXCHANGES_BEFORE_SUMMARY

    user_id = "e2e_summary_test"
    mm = MemoryManager(tmp_session_dir)
    session = mm.load_session(user_id)

    # Simulate 25 exchanges (above main's MAX_EXCHANGES_BEFORE_SUMMARY = 20)
    for i in range(25):
        session.exchanges.append((f"user msg {i}", f"bot reply {i}", None))
    session.exchange_count = 25

    reason_main = mm.should_summarize(session)
    check(
        f"main: should_summarize() triggers at exchange_count=25 (threshold={MAX_EXCHANGES_BEFORE_SUMMARY})",
        bool(reason_main),
        f"reason='{reason_main}' (empty = no trigger)",
    )

    if features["token_based_summary"]:
        try:
            # gpt-oss has 128k context; threshold = 128k * 0.6 ≈ 76k tokens
            # 25 short exchanges ≈ 3k tokens (far below) → no summary
            reason_low = mm.should_summarize(session, summary_threshold_tokens=76_800)
            check(
                "branch: 25 exchanges / ~3k tokens / 76k threshold → no token-based summary",
                "token" not in reason_low,
                f"reason='{reason_low}' (exchange_limit is expected since count=25 > 20)",
            )
            # exchange_count trigger still fires (25 >= MAX_EXCHANGES_BEFORE_SUMMARY=20)
            check(
                "branch: exchange_count trigger still active at 25 exchanges",
                bool(reason_low),
                f"reason='{reason_low}' (should be 'exchange_limit')",
            )

            # Simulate a session with heavy history (>76k tokens worth)
            session_heavy = mm.load_session("e2e_heavy_session")
            big_text = "x" * (76_800 * 4 + 100)  # chars → ~76.8k+ tokens
            session_heavy.daily_history = big_text
            session_heavy.exchange_count = 5  # below count threshold
            reason_heavy = mm.should_summarize(session_heavy, summary_threshold_tokens=76_800)
            check(
                "branch: 5 exchanges but heavy history (>76k tokens) → token summary triggers",
                "token" in reason_heavy,
                f"reason='{reason_heavy}'",
            )
        except Exception as exc:
            check("branch: token-based summarization", False, str(exc))
    else:
        check(
            "branch: token-based summarization available",
            False,
            "main — fixed count only (no summary_threshold_tokens parameter)",
        )

    # ── With real API: run 22 exchanges and confirm memory retention ──────────
    section("Test 2b — Memory retention after >20 exchanges (real API)")
    user_id2 = "e2e_summary_live"
    ws2 = os.path.join(tmp_session_dir, user_id2, "workspace")
    os.makedirs(ws2, exist_ok=True)
    for fname, content in [
        ("identity.md", "# PawLia\nDu bist PawLia, ein hilfreicher Assistent.\n"),
        ("soul.md", "## Charakter\nFreundlich, präzise.\n"),
        ("user.md", "## Nutzer\nTestnutzer.\n"),
    ]:
        with open(os.path.join(ws2, fname), "w", encoding="utf-8") as f:
            f.write(content)
    agent = app.make_agent(user_id2)

    # Plant a fact early
    _ = await agent.run("Merk dir bitte: mein Lieblingstier ist ein Axolotl.")
    _ = await agent.run("Was ist 2+2?")
    # Pad up to 22 exchanges with trivial questions
    for k in range(20):
        _ = await agent.run(f"Was ist {k+3}+{k+3}?")

    answer = await agent.run("Was ist mein Lieblingstier?")
    _p(f"  {_DIM}Nach 22 Turns:{_RESET} {answer[:200]}")
    check(
        "Axolotl nach 22 Turns noch erinnerlich",
        "axolotl" in answer.lower(),
        f"Antwort: {answer[:300]}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Agent override API
# ──────────────────────────────────────────────────────────────────────────────

def test_agent_overrides(app, tmp_session_dir: str):
    section("Test 3 — Agent Override API")

    mm = app.memory
    session = mm.load_session("e2e_override_test")

    # set_agent_override_value / get_agent_override_value should exist (added as shim)
    has_set = hasattr(mm, "set_agent_override_value")
    has_get = hasattr(mm, "get_agent_override_value")
    has_eff = hasattr(mm, "effective_agent_overrides")
    check("MemoryManager.set_agent_override_value vorhanden", has_set)
    check("MemoryManager.get_agent_override_value vorhanden", has_get)
    check("MemoryManager.effective_agent_overrides vorhanden", has_eff)

    if has_set and has_get:
        mm.set_agent_override_value(session, "chat", "gpt-oss")
        val = mm.get_agent_override_value(session, "chat")
        check("Override 'chat' setzen und lesen", val == "gpt-oss", f"got '{val}'")

        mm.set_agent_override_value(session, "chat", None)
        val_cleared = mm.get_agent_override_value(session, "chat")
        check("Override 'chat' löschen", val_cleared is None, f"got '{val_cleared}'")

    # LLMFactory.default_model_name
    has_default_model = hasattr(app.llm, "default_model_name")
    check("LLMFactory.default_model_name vorhanden", has_default_model)
    if has_default_model:
        name = app.llm.default_model_name("chat")
        check(
            f"default_model_name('chat') gibt einen Key zurück ('{name}')",
            bool(name),
            f"got '{name}'",
        )
        # With override
        override_name = app.llm.default_model_name("chat", agent_overrides={"chat": "gpt-oss"})
        check(
            "default_model_name mit agent_override=gpt-oss",
            override_name == "gpt-oss",
            f"got '{override_name}'",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: App reload
# ──────────────────────────────────────────────────────────────────────────────

def test_app_reload(app):
    section("Test 4 — App.reload()")

    has_reload = hasattr(app, "reload")
    check("app.reload() vorhanden", has_reload)
    if not has_reload:
        return

    old_llm_id = id(app.llm)
    old_skills = set(app._bundled_skills.keys())
    result = app.reload()
    check("reload() gibt dict zurück", isinstance(result, dict), str(type(result)))
    check("reload() enthält config_path", "config_path" in result)
    check("reload() enthält model_count", "model_count" in result)
    check("LLMFactory nach reload neu instanziiert", id(app.llm) != old_llm_id)
    new_skills = set(app._bundled_skills.keys())
    check("Skills nach reload konsistent", new_skills == old_skills,
          f"vorher={sorted(old_skills)}, nachher={sorted(new_skills)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    try:
        import subprocess
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        branch = "unknown"

    _p(f"\n{_BOLD}{'═'*65}{_RESET}")
    _p(f"{_BOLD}  PawLia Memory System — E2E Vergleichstest{_RESET}")
    _p(f"{_BOLD}  Branch: {_CYAN}{branch}{_RESET}")
    _p(f"{_BOLD}{'═'*65}{_RESET}")

    features = _detect_branch_features()
    _p(f"\n  Feature-Erkennung:")
    for k, v in features.items():
        icon = _GREEN + "✓" + _RESET if v else _RED + "✗" + _RESET
        _p(f"    {icon}  {k}")

    tmp = tempfile.mkdtemp(prefix="pawlia_e2e_")
    try:
        from pawlia.app import create_app
        app = create_app()
        # Override session dir to temp dir for isolation.
        # app.session_dir is used by make_runner (→ files skill), so both must match.
        app.session_dir = tmp
        app.memory.session_dir = tmp
        app.memory._sessions = {}

        await test_workspace_search(app, tmp, features)
        await test_summarization_threshold(app, tmp, features)
        test_agent_overrides(app, tmp)
        test_app_reload(app)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _p(f"\n{_BOLD}{'═'*65}{_RESET}")
    _p(f"{_BOLD}  Ergebnisse  ({_GREEN}{_PASS} PASS{_RESET}{_BOLD}"
       f" · {_RED}{_FAIL} FAIL{_RESET}{_BOLD}"
       f" · {_DIM}{_SKIP} SKIP{_RESET}{_BOLD}){_RESET}")
    _p(f"{_BOLD}{'═'*65}{_RESET}")

    if _FAIL:
        _p(f"\n{_RED}Fehlgeschlagene Tests:{_RESET}")
        for status, name, detail in _results:
            if status == "FAIL":
                _p(f"  • {name}")
                if detail:
                    _p(f"    {_DIM}{detail[:200]}{_RESET}")

    return _FAIL


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
