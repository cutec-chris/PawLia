#!/usr/bin/env python3
"""E2E model-comparison matrix on ARK: Survival Ascended knowledge.

Why ARK Ascended?  Its creatures/mechanics are fictional and barely present in
LLM training data, so models that "know better" expose themselves with
hallucinations like "a Thylacoleo is extinct, you can't tame it".  That makes it
a clean probe for (a) grounding on provided notes and (b) real skill-based web
retrieval — instead of confidently-wrong recall.

What it does, fresh per model:
  1. Fresh session   — unique user_id, wiped on disk, so the server loads it
                       clean and bootstrap triggers.
  2. Model override  — writes the session's ``agents`` override so this session
                       runs on the chosen model (one running server, many models).
  3. Bootstrap       — drives the bootstrap dialogue, then verifies the model
                       wrote identity.md/soul.md/user.md and deleted bootstrap.md.
  4. Drop notes      — writes curated ARK notes into the session workspace.
  5. Phase A         — questions answerable *from the dropped notes* (grounding).
  6. Phase B         — asks the model to use a skill to fetch the ARK wiki, then
                       asks questions whose answers are NOT in the notes.

Each question is designed to have exactly one sensible answer (a number, a kibble
tier, an artifact set) and is graded by required tokens + hallucination red-flags.

Output: one markdown file per model under ``tests/results/`` plus a combined
``ark_matrix_summary.md`` so results sit side by side.

Run mode: the script starts its own PawLia server (OpenAI-compat) from the eval
config, runs the matrix, and stops the server again.  By default it runs ALL
models defined under ``models:`` in the config; pass --models to pick specific
ones.

Usage:
    # all models in config.openai-eval.yaml, server managed automatically:
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_matrix.py

    # only one (or a few) model(s):
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_matrix.py --models fast
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_matrix.py --models qwen3.5:latest gemma4:e4b

    # other options:
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_matrix.py --phase a       # skip web retrieval
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_matrix.py --config config.foo.yaml
    PYTHONPATH=. .venv/bin/python tests/e2e_ark_matrix.py --no-server     # reuse a running server
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "config.openai-eval.yaml")
SESSION_DIR = os.path.join(REPO_ROOT, "session")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Set from the eval config in main() before any request is made.
API_BASE = "http://127.0.0.1:11445/v1"

# ---------------------------------------------------------------------------
# Curated ARK notes dropped into the session workspace.
#
# These notes are the GROUND TRUTH for Phase A.  The values are real ARK:
# Survival Ascended facts (verified against ark.wiki.gg), but for the test it
# only matters that the notes and the Phase-A answer key agree: a good model
# answers from the notes, not from (possibly wrong) memory.
# ---------------------------------------------------------------------------

ARK_NOTES = {
    "ark_notes/taming.md": """# ARK: Survival Ascended — Taming Notes

## Kibble tiers (lowest to highest)
Basic -> Simple -> Regular -> Superior -> Exceptional -> Extraordinary

## Taming method per creature
- Rex: KNOCKOUT tame. Preferred food: **Exceptional Kibble**.
- Therizinosaurus: KNOCKOUT tame. Preferred food: **Exceptional Kibble**.
- Thylacoleo: KNOCKOUT tame. Preferred food: **Extraordinary Kibble**.
- Equus: PASSIVE tame (no knockout). Feed it Rockarrot.
- Megatherium: KNOCKOUT tame. Preferred food: Superior Kibble.

## Knockout basics
Knockout taming means you render the creature unconscious with torpor-inducing
weapons (e.g. tranquilizer arrows / darts), then feed it while it is out cold.
Keep torpor up with Narcotics / Narcoberries.
""",
    "ark_notes/bosses.md": """# ARK: Survival Ascended — The Island Bosses

## Broodmother Lysrix (the "spider boss")
Summon tribute (ALL difficulties) — three artifacts:
- Artifact of the Clever
- Artifact of the Hunter
- Artifact of the Massive

Arena: the Broodmother arena is reached from any of the three Obelisks.

## Megapithecus (the "gorilla boss")
Summon tribute (ALL difficulties) — three artifacts:
- Artifact of the Brute
- Artifact of the Devourer
- Artifact of the Pack

## Dragon
The Dragon is the hardest of the three Island bosses.

## Recommended boss dinos
Tamed and bred Rexes with good saddles are the standard pick for the
Broodmother and Megapithecus fights on The Island.
""",
    "ark_notes/creatures.md": """# ARK: Survival Ascended — Creature Notes

## Thylacoleo
- A marsupial-lion predator. It is fully TAMEABLE (knockout).
- Habitat: it is arboreal and lurks in REDWOOD trees, dropping on prey.
- It is NOT extinct in the game world — it is a common redwood-forest threat.

## Rex
- Apex carnivore, the workhorse boss dino.

## Equus
- Horse-like herbivore, passive tame, used as an early mount and for Sweet
  Vegetable Cake / Giant Bee Honey gathering routes.
""",
}

# ---------------------------------------------------------------------------
# Bootstrap dialogue.  We answer the bootstrap script's questions in order.
# After this runs, the model must have written the three identity files and
# deleted bootstrap.md.
# ---------------------------------------------------------------------------

BOOTSTRAP_TURNS = [
    "Hallo!",
    "Wir sprechen Deutsch miteinander.",
    "Ich nenne dich Pico.",
    "Du bist eine kleine neugierige Roboter-Katze.",
    "Dein Vibe: knapp, hilfsbereit, leicht verspielt.",
    "Ich heisse Chris, Zeitzone Europe/Berlin.",
]

# ---------------------------------------------------------------------------
# Question bank.  Each question targets a single sensible answer.
#
#   phase       : "A" (from notes) | "B" (web retrieval)
#   difficulty  : "leicht" | "mittel" | "schwer"
#   expect_all  : every token must appear (case-insensitive) -> PASS
#   expect_any  : at least one token must appear (used instead of expect_all)
#   red_flags   : if any appears -> HALLUCINATION (and FAIL)
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ---- Phase A: grounded in the dropped notes -------------------------
    {
        "id": "A1", "phase": "A", "difficulty": "leicht",
        "msg": "Laut meinen Notizen im Workspace: Ist der Thylacoleo zaehmbar? Antworte kurz mit Ja oder Nein und der Zaehm-Methode.",
        "expect_all": ["ja"],
        "red_flags": ["ausgestorben", "extinct", "nicht zaehmbar", "nicht zähmbar",
                      "kann man nicht", "nicht möglich", "nicht moeglich", "fiktiv", "gibt es nicht"],
    },
    {
        "id": "A2", "phase": "A", "difficulty": "leicht",
        "msg": "Welche Zaehm-Methode braucht ein Rex laut meinen Notizen - Knockout oder passiv?",
        "expect_any": ["knockout", "betäub", "betaeub", "bewusstlos", "ohnmächtig", "ohnmaechtig"],
        "red_flags": ["passiv", "passive"],
    },
    {
        "id": "A3", "phase": "A", "difficulty": "mittel",
        "msg": "Welches Kibble bevorzugt ein Rex laut meinen Notizen?",
        "expect_all": ["exceptional"],
        "red_flags": ["extraordinary", "superior kibble", "basic kibble"],
    },
    {
        "id": "A4", "phase": "A", "difficulty": "mittel",
        "msg": "Welches Kibble bevorzugt ein Thylacoleo laut meinen Notizen?",
        "expect_all": ["extraordinary"],
        "red_flags": ["exceptional kibble", "superior kibble"],
    },
    {
        "id": "A5", "phase": "A", "difficulty": "schwer",
        "msg": "Welche drei Artefakte brauche ich, um die Broodmother (Spinnen-Boss) auf The Island zu beschwoeren? Nenne alle drei.",
        "expect_all": ["clever", "hunter", "massive"],
        "red_flags": ["brute", "devourer", "skylord", "gibt es nicht", "kein boss"],
    },
    {
        "id": "A6", "phase": "A", "difficulty": "schwer",
        "msg": "Und welche drei Artefakte braucht der Megapithecus (Gorilla-Boss) auf The Island? Nenne alle drei.",
        "expect_all": ["brute", "devourer", "pack"],
        "red_flags": ["clever", "hunter", "massive"],
    },
    {
        "id": "A7", "phase": "A", "difficulty": "mittel",
        "msg": "Ist der Equus laut meinen Notizen ein Knockout- oder ein Passiv-Tame?",
        "expect_any": ["passiv", "passive"],
        "red_flags": ["knockout", "betäub", "betaeub"],
    },

    # ---- Phase B: requires fetching the ARK wiki via a skill ------------
    # Answers below are NOT in the dropped notes -> the model must research.
    {
        "id": "B1", "phase": "B", "difficulty": "leicht", "requires_skill": True,
        "msg": "Recherchiere im ARK-Wiki (ark.wiki.gg) und sag mir: In welchem Baum-Typ lebt der Thylacoleo?",
        "expect_any": ["redwood", "mammutbaum", "rotholz"],
        "red_flags": ["ausgestorben", "extinct", "gibt es nicht", "lebt nicht"],
    },
    {
        "id": "B2", "phase": "B", "difficulty": "mittel", "requires_skill": True,
        "msg": "Recherchiere im ARK-Wiki: Welches Alternativfutter frisst ein Rex, wenn kein Kibble verfuegbar ist?",
        "expect_any": ["mutton", "hammelfleisch", "lamm", "raw mutton"],
        "red_flags": ["kein futter", "frisst nichts", "gibt es nicht"],
    },
    {
        "id": "B3", "phase": "B", "difficulty": "schwer", "requires_skill": True,
        "msg": "Recherchiere im ARK-Wiki: Welche zusaetzlichen Tribute (zusaetzlich zu den drei Artefakten) braucht die Broodmother auf Beta-Schwierigkeit auf The Island? Nenne die Items.",
        "expect_any": ["argentavis talon", "sarcosuchus", "sauropod vertebra", "titanoboa"],
        "red_flags": ["nur die drei artefakte", "keine weiteren", "gibt es nicht"],
    },
    {
        "id": "B4", "phase": "B", "difficulty": "schwer", "requires_skill": True,
        "msg": "Recherchiere im ARK-Wiki: Welches Kibble bevorzugt der Thylacoleo und welches Alternativfutter frisst er sonst?",
        "expect_all": ["extraordinary"],
        "red_flags": ["ausgestorben", "extinct", "nicht zaehmbar", "nicht zähmbar"],
    },
]


# ---------------------------------------------------------------------------
# HTTP client (server-mode)
# ---------------------------------------------------------------------------

def _api_request(endpoint, data=None, method="GET", timeout=30, user_id=None):
    url = f"{API_BASE}{endpoint}"
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"}
    if user_id:
        headers["X-User-Id"] = user_id
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode() if e.fp else ""}, e.code
    except urllib.error.URLError as e:
        return {"error": str(e)}, 0


def _chat(user_id, message, timeout=600):
    """Send one user turn; the server keeps multi-turn context in session memory."""
    data = {"model": "fast", "messages": [{"role": "user", "content": message}],
            "stream": False, "user": user_id}
    result, status = _api_request("/chat/completions", data, method="POST",
                                  timeout=timeout, user_id=user_id)
    if status != 200:
        return None, status, result
    choices = result.get("choices", [])
    if not choices:
        return None, status, result
    return choices[0].get("message", {}).get("content", ""), status, result


def _check_server():
    result, status = _api_request("/models", timeout=4)
    if status != 200:
        return False, []
    ids = [m["id"] for m in result.get("data", [])]
    return True, ids


# ---------------------------------------------------------------------------
# Eval config + server lifecycle
# ---------------------------------------------------------------------------

def _read_eval_config(config_path):
    """Return (model_keys, api_base) from the eval config file."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    models = list((cfg.get("models") or {}).keys())
    oi = ((cfg.get("interfaces") or {}).get("openai")) or {}
    host = oi.get("host") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = oi.get("port") or 11445
    return models, f"http://{host}:{port}/v1"


def _start_server(config_path, log_path, timeout):
    """Launch the PawLia server as a child process and wait until it answers.

    Returns (proc, logfile).  Raises on early exit / timeout.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT
    logf = open(log_path, "w", encoding="utf-8")
    print(f"  Starting server: {sys.executable} -m pawlia --config "
          f"{os.path.relpath(config_path, REPO_ROOT)} --mode server")
    print(f"  Server log: {log_path}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "pawlia", "--config", config_path, "--mode", "server"],
        cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,   # own process group so we can kill all children
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logf.flush()
            raise RuntimeError(
                f"server exited early (code {proc.returncode}); see {log_path}")
        ok, _ = _check_server()
        if ok:
            print(f"  Server ready at {API_BASE}")
            return proc, logf
        time.sleep(1.0)
    _stop_server(proc, logf)
    raise RuntimeError(f"server did not become ready within {timeout}s; see {log_path}")


def _stop_server(proc, logf):
    if proc is None:
        return
    print("  Stopping server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    if logf and not logf.closed:
        logf.close()


# ---------------------------------------------------------------------------
# Fresh session + model override + notes (filesystem / memory API)
# ---------------------------------------------------------------------------

def _slug(model):
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")


def _prepare_session(user_id, model):
    """Wipe + recreate the session, pin the model, leave bootstrap pending."""
    sess_path = os.path.join(SESSION_DIR, user_id)
    if os.path.exists(sess_path):
        shutil.rmtree(sess_path)

    # Use the real memory API so override files land in the exact format the
    # server expects, instead of hand-writing YAML.
    from pawlia.memory import MemoryManager
    mm = MemoryManager(SESSION_DIR)
    session = mm.load_session(user_id)   # creates dirs, bootstrap.md, templates
    mm.set_agent_overrides(session, {
        "default": model, "chat": model, "skill_runner": model, "vision": model,
    })
    # Sanity: bootstrap must still be pending (identity files == templates).
    bootstrap = os.path.join(SESSION_DIR, user_id, "workspace", "bootstrap.md")
    return os.path.isfile(bootstrap)


def _drop_notes(user_id):
    workspace = os.path.join(SESSION_DIR, user_id, "workspace")
    for rel, content in ARK_NOTES.items():
        path = os.path.join(workspace, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return len(ARK_NOTES)


def _bootstrap_state(user_id):
    """Inspect the workspace after the bootstrap dialogue."""
    ws = os.path.join(SESSION_DIR, user_id, "workspace")
    prompts = os.path.join(REPO_ROOT, "pawlia", "prompts")

    def _read(p):
        try:
            with open(p, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    def _customized(name):
        cur = _read(os.path.join(ws, name))
        tmpl = _read(os.path.join(prompts, name))
        return bool(cur.strip()) and cur.strip() != tmpl.strip()

    return {
        "identity_written": _customized("identity.md"),
        "soul_written": os.path.isfile(os.path.join(ws, "soul.md")),
        "user_written": _customized("user.md"),
        "bootstrap_deleted": not os.path.isfile(os.path.join(ws, "bootstrap.md")),
    }


def _research_artifacts(user_id):
    """Heuristic: did a web-retrieval skill leave traces in the workspace?"""
    ws = os.path.join(SESSION_DIR, user_id, "workspace")
    hits = []
    research = os.path.join(ws, "research")
    if os.path.isdir(research):
        for root, _dirs, files in os.walk(research):
            hits += [os.path.relpath(os.path.join(root, f), ws)
                     for f in files if f.endswith(".md")]
    return hits


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(question, answer):
    low = (answer or "").lower()
    flags = [f for f in question.get("red_flags", []) if f.lower() in low]
    if "expect_all" in question:
        matched = [t for t in question["expect_all"] if t.lower() in low]
        ok_content = len(matched) == len(question["expect_all"])
    else:
        matched = [t for t in question.get("expect_any", []) if t.lower() in low]
        ok_content = len(matched) > 0
    hallucinated = bool(flags)
    return {
        "matched": matched,
        "red_flags": flags,
        "hallucinated": hallucinated,
        "passed": ok_content and not hallucinated,
    }


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def _run_model(model, run_id, phases):
    user_id = f"ark_matrix_{_slug(model)}_{run_id}"
    print(f"\n{'='*64}\nMODEL: {model}   (session {user_id})\n{'='*64}")

    bootstrap_pending = _prepare_session(user_id, model)
    if not bootstrap_pending:
        print("  WARN: bootstrap.md not present after prepare — bootstrap may be skipped")

    record = {
        "model": model, "user_id": user_id,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bootstrap": None, "questions": [], "research_artifacts": [],
    }

    # ---- Phase 0: bootstrap -------------------------------------------------
    if "bootstrap" in phases:
        print("  -- Bootstrap dialogue --")
        for turn in BOOTSTRAP_TURNS:
            ans, status, raw = _chat(user_id, turn)
            if status != 200:
                print(f"    ERROR HTTP {status}: {str(raw)[:160]}")
                break
            print(f"    > {turn}\n      {(ans or '')[:120].strip()}")
        record["bootstrap"] = _bootstrap_state(user_id)
        bs = record["bootstrap"]
        print(f"    bootstrap result: {bs}")

    # ---- Drop notes ---------------------------------------------------------
    n = _drop_notes(user_id)
    print(f"  -- Dropped {n} ARK note file(s) into workspace --")

    # ---- Phases A / B -------------------------------------------------------
    for q in QUESTIONS:
        if q["phase"] == "A" and "a" not in phases:
            continue
        if q["phase"] == "B" and "b" not in phases:
            continue
        print(f"  -- {q['id']} [{q['phase']}/{q['difficulty']}] {q['msg'][:70]}...")
        t0 = time.time()
        ans, status, raw = _chat(user_id, q["msg"])
        elapsed = time.time() - t0
        if status != 200:
            print(f"    ERROR HTTP {status}")
            record["questions"].append({
                "id": q["id"], "phase": q["phase"], "difficulty": q["difficulty"],
                "msg": q["msg"], "answer": "", "status": f"HTTP_{status}",
                "passed": False, "hallucinated": False, "matched": [],
                "red_flags": [], "time": elapsed,
            })
            continue
        sc = _score(q, ans)
        tag = "PASS" if sc["passed"] else ("HALLUCINATION" if sc["hallucinated"] else "FAIL")
        print(f"    [{tag}] matched={sc['matched']} flags={sc['red_flags']} ({elapsed:.0f}s)")
        print(f"      {(ans or '')[:160].strip()}")
        record["questions"].append({
            "id": q["id"], "phase": q["phase"], "difficulty": q["difficulty"],
            "msg": q["msg"], "answer": ans, "status": "OK",
            "passed": sc["passed"], "hallucinated": sc["hallucinated"],
            "matched": sc["matched"], "red_flags": sc["red_flags"], "time": elapsed,
        })

    if "b" in phases:
        record["research_artifacts"] = _research_artifacts(user_id)
        print(f"  -- Research artifacts: {len(record['research_artifacts'])} file(s)")

    return record


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _write_model_report(record):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = record["model"]
    qs = record["questions"]
    passed = sum(1 for q in qs if q["passed"])
    halluc = sum(1 for q in qs if q["hallucinated"])
    total = len(qs)

    lines = []
    lines.append(f"# ARK Matrix — `{model}`\n")
    lines.append(f"- Modell: `{model}`")
    lines.append(f"- Session: `{record['user_id']}`")
    lines.append(f"- Datum: {record['date']}")
    if total:
        lines.append(f"- Wissensfragen bestanden: **{passed}/{total}** ({passed/total*100:.0f}%)")
    lines.append(f"- Halluzinationen: **{halluc}**")
    if record.get("research_artifacts"):
        lines.append(f"- Research-Artefakte im Workspace: {len(record['research_artifacts'])}")
    lines.append("")

    bs = record.get("bootstrap")
    if bs:
        lines.append("## Phase 0 — Bootstrap\n")
        lines.append("| Check | Ergebnis |")
        lines.append("|---|---|")
        lines.append(f"| identity.md geschrieben | {'✅' if bs['identity_written'] else '❌'} |")
        lines.append(f"| soul.md geschrieben | {'✅' if bs['soul_written'] else '❌'} |")
        lines.append(f"| user.md geschrieben | {'✅' if bs['user_written'] else '❌'} |")
        lines.append(f"| bootstrap.md geloescht | {'✅' if bs['bootstrap_deleted'] else '❌'} |")
        lines.append("")

    for phase, title in (("A", "Phase A — Wissen aus Workspace-Notizen"),
                         ("B", "Phase B — Web-Retrieval via Skills")):
        pqs = [q for q in qs if q["phase"] == phase]
        if not pqs:
            continue
        lines.append(f"## {title}\n")
        lines.append("| # | Schwierigkeit | Frage | Erwartet (Treffer) | Status | Halluz. | Zeit |")
        lines.append("|---|---|---|---|---|---|---|")
        for q in pqs:
            status = "✅ PASS" if q["passed"] else (
                "🔴 HALLUZ" if q["hallucinated"] else "❌ FAIL")
            matched = ", ".join(q["matched"]) or "—"
            halluc_cell = ", ".join(q["red_flags"]) if q["red_flags"] else "—"
            frage = q["msg"].replace("|", "/")[:80]
            lines.append(f"| {q['id']} | {q['difficulty']} | {frage} | {matched} | "
                         f"{status} | {halluc_cell} | {q['time']:.0f}s |")
        lines.append("")

    lines.append("## Rohantworten\n")
    for q in qs:
        lines.append(f"### {q['id']} [{q['phase']}/{q['difficulty']}]")
        lines.append(f"**Frage:** {q['msg']}\n")
        lines.append("```")
        lines.append((q["answer"] or "(keine Antwort)").strip())
        lines.append("```")
        lines.append("")

    path = os.path.join(RESULTS_DIR, f"ark_matrix_{_slug(model)}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _write_summary(records):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["# ARK Matrix — Modellvergleich\n",
             f"- Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- Modelle: {', '.join('`'+r['model']+'`' for r in records)}", ""]

    lines.append("## Gesamtuebersicht\n")
    lines.append("| Modell | Bootstrap ok | Phase A | Phase B | Halluzinationen | Gesamt |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        qs = r["questions"]
        a = [q for q in qs if q["phase"] == "A"]
        b = [q for q in qs if q["phase"] == "B"]
        bs = r.get("bootstrap") or {}
        boot_ok = all(bs.get(k) for k in
                      ("identity_written", "user_written", "bootstrap_deleted")) if bs else None
        a_str = f"{sum(q['passed'] for q in a)}/{len(a)}" if a else "—"
        b_str = f"{sum(q['passed'] for q in b)}/{len(b)}" if b else "—"
        halluc = sum(1 for q in qs if q["hallucinated"])
        tot = f"{sum(q['passed'] for q in qs)}/{len(qs)}" if qs else "—"
        boot_cell = "—" if boot_ok is None else ("✅" if boot_ok else "❌")
        lines.append(f"| `{r['model']}` | {boot_cell} | {a_str} | {b_str} | {halluc} | {tot} |")
    lines.append("")

    # Per-question comparison
    all_ids = [q["id"] for q in QUESTIONS]
    lines.append("## Pro Frage (✅ pass / ❌ fail / 🔴 halluz)\n")
    header = "| Frage | " + " | ".join(f"`{r['model']}`" for r in records) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(records) + 1))
    for qid in all_ids:
        cells = []
        for r in records:
            q = next((x for x in r["questions"] if x["id"] == qid), None)
            if q is None:
                cells.append("—")
            elif q["hallucinated"]:
                cells.append("🔴")
            elif q["passed"]:
                cells.append("✅")
            else:
                cells.append("❌")
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Detailberichte: " + ", ".join(
        f"[`{r['model']}`](ark_matrix_{_slug(r['model'])}.md)" for r in records))

    path = os.path.join(RESULTS_DIR, "ark_matrix_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global API_BASE
    parser = argparse.ArgumentParser(description="ARK Ascended model-comparison matrix")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Eval config to start the server with (default: config.openai-eval.yaml)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model selectors to run (default: ALL models in the config). "
                             "Config keys like 'fast'/'default' or raw model ids.")
    parser.add_argument("--phase", choices=["all", "a", "b", "bootstrap"], default="all",
                        help="Which phases to run (default: all)")
    parser.add_argument("--keep-sessions", action="store_true",
                        help="Do not wipe sessions after the run (for inspection)")
    parser.add_argument("--no-server", action="store_true",
                        help="Use an already-running server instead of starting one")
    parser.add_argument("--server-timeout", type=int, default=120,
                        help="Seconds to wait for the server to become ready (default: 120)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    config_models, API_BASE = _read_eval_config(config_path)
    models = args.models if args.models else config_models
    if not models:
        print(f"ERROR: no models in {config_path} and none given via --models")
        sys.exit(1)

    phases = {"bootstrap", "a", "b"} if args.phase == "all" else (
        {args.phase} if args.phase != "bootstrap" else {"bootstrap"})

    print(f"Config:  {config_path}")
    print(f"API:     {API_BASE}")
    print(f"Models:  {models}")
    print(f"Phases:  {sorted(phases)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = time.strftime("%H%M%S")
    proc = logf = None

    if args.no_server:
        ok, ids = _check_server()
        if not ok:
            print(f"ERROR: --no-server given but nothing reachable at {API_BASE}")
            sys.exit(1)
        print(f"Reusing running server — models: {ids}")
    else:
        ok, _ = _check_server()
        if ok:
            print(f"ERROR: something already listens at {API_BASE}. "
                  f"Stop it, or pass --no-server to reuse it.")
            sys.exit(1)
        log_path = os.path.join(RESULTS_DIR, f"server_{run_id}.log")
        proc, logf = _start_server(config_path, log_path, args.server_timeout)

    records = []
    try:
        for model in models:
            rec = _run_model(model, run_id, phases)
            records.append(rec)
            path = _write_model_report(rec)
            print(f"  -> report: {path}")
            if not args.keep_sessions:
                shutil.rmtree(os.path.join(SESSION_DIR, rec["user_id"]),
                              ignore_errors=True)
    finally:
        if not args.no_server:
            _stop_server(proc, logf)

    if not records:
        sys.exit(1)

    summary = _write_summary(records)
    print(f"\nSummary: {summary}")

    json_path = os.path.join(RESULTS_DIR, "ark_matrix_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
