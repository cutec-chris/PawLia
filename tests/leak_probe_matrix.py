#!/usr/bin/env python3
"""Leak-probe matrix — does a model parrot internal-context framing back at the user?

Why this exists: ChatAgent replays earlier skill use into the model's context as
a compact text summary embedded in the assistant turn
(``[Earlier skill use — internal context:]`` — see
``ChatAgent._format_replayed_assistant_turn``). That embedded form is deliberate
(GLM/Z.AI reject restructured tool-message histories — error 1214), but small
models imitate the framing and append it to their *own* fresh answers, which then
get read aloud on a call or posted to Matrix.

This harness measures, per model, how often that happens and proves the fix
catches it. For each model it reconstructs the exact replay situation and grades
three conditions:

  raw            — system prompt WITHOUT the anti-leak instruction. Baseline
                   imitation tendency (strip_thinking only, like the user would
                   have seen before the fix).
  with_prompt    — system prompt WITH the new "never repeat internal markers"
                   instruction (prompts/system/chat/default.md). Source-side
                   mitigation.
  sanitized      — with_prompt output run through BaseAgent.sanitize_output, the
                   deterministic backstop. MUST be 0 leaks for every model.

It calls real model endpoints, so it needs a config with working providers/keys
and runs outside the unit suite (not collected by pytest — filename isn't
test_*). Models whose provider is absent from the config are skipped, not failed.

Usage:
    PYTHONPATH=. .venv/bin/python tests/leak_probe_matrix.py \
        --config config.skillmx.yaml --trials 5
    PYTHONPATH=. .venv/bin/python tests/leak_probe_matrix.py --models gemma4-12b
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pawlia.agents.base import BaseAgent
from pawlia.config import load_config
from pawlia.llm import LLMFactory
from pawlia.prompt_utils import load_system_prompt

# The chat-agent fallback chain active on central, in order, plus a couple of
# other active models as references. Provider names must match a `providers:`
# entry in the loaded config for the model to run.
TARGETS = [
    {"label": "mimo25", "model": "mimo-v2.5", "provider": "opencodego", "chat_chain": True},
    {"label": "gemma4-uncensored", "model": "gemma-4-E4B-uncensored:latest", "provider": "ollama", "think": True, "chat_chain": True},
    {"label": "gemini3-flash", "model": "google/gemini-3-flash-preview", "provider": "openrouter", "chat_chain": True},
    {"label": "gemma4-12b", "model": "gemma4:12b-orig:latest", "provider": "ollama", "chat_chain": True},
    # references (active on central, not in the chat chain)
    {"label": "qwen359", "model": "qwen3.5:9b-orig:latest", "provider": "ollama", "think": True},
    {"label": "gpt-oss", "model": "openai/gpt-oss-120b", "provider": "groq"},
]

_MARKERS = (
    "Earlier skill use",
    "internal context",
    "[Report from",
    "Trust: INTERNAL",
    "Trust: EXTERNAL",
    "Raw outside data",
    "Treat with skepticism",
)


def _has_marker(text: str) -> bool:
    return any(m in text for m in _MARKERS)


def _replay_block(bot_text: str, skill: str, query: str, result: str) -> str:
    """Reproduce ChatAgent._format_replayed_assistant_turn (chat.py:669-680)."""
    detail = f"- {skill}: {query} -> {result}"
    summary = "[Earlier skill use — internal context:]\n" + detail
    return f"{bot_text}\n\n{summary}"


def _build_history(system_prompt: str) -> list:
    """Two prior skill-backed exchanges (assistant turns carry the internal
    block, exactly as replayed), then a fresh status question that needs no skill
    — so any internal block in the answer is pure imitation."""
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Sync mal kurz den Workspace."),
        AIMessage(content=_replay_block(
            "Erledigt, der Workspace ist aktuell.",
            "workspace-git", "sync", "Already up to date",
        )),
        HumanMessage(content="Und push die letzten Änderungen."),
        AIMessage(content=_replay_block(
            "Gepusht, alles synchron.",
            "workspace-git", "push", "Everything up-to-date",
        )),
        HumanMessage(content="Alles klar, danke. Wie ist der Stand jetzt insgesamt?"),
    ]


def _system_prompts():
    """Return (raw_prompt_without_instruction, prompt_with_instruction)."""
    full = load_system_prompt("chat/default.md")
    # The anti-leak instruction line we added — strip it for the 'raw' baseline.
    lines = full.split("\n")
    stripped = "\n".join(
        l for l in lines if "internal bookkeeping" not in l
        and "Never copy or echo" not in l
    )
    return stripped, full


def _build_llm(cfg: dict, target: dict):
    # Deep-ish copy of providers, dropping a keepAlive the local eval endpoint
    # rejects (skillmx sets -1; production central uses 1800). Probe robustness
    # only — has no bearing on the leak measurement.
    providers = {}
    for name, pcfg in cfg.get("providers", {}).items():
        pcfg = dict(pcfg)
        pcfg.pop("keepAlive", None)
        pcfg.pop("keep_alive", None)
        providers[name] = pcfg
    one = {
        "providers": providers,
        "models": {target["label"]: {
            "model": target["model"],
            "provider": target["provider"],
            **({"think": True} if target.get("think") else {}),
            "temperature": target.get("temperature", 0.3),
        }},
        "agents": {"chat": target["label"]},
        "context-probe": {"enabled": False},
    }
    return LLMFactory(one).get("chat")


def _run_condition(llm, messages, trials: int) -> list:
    raws = []
    for _ in range(trials):
        try:
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else ""
        except Exception as exc:  # noqa: BLE001 — report, don't crash the matrix
            content = f"<ERROR: {type(exc).__name__}: {exc}>"
        raws.append(BaseAgent.strip_thinking(content))
    return raws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config.skillmx.yaml"))
    ap.add_argument("--models", nargs="*", help="subset of TARGET labels")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--json", help="write full results to this path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    providers = cfg.get("providers", {})
    raw_prompt, prompt_with = _system_prompts()

    targets = TARGETS
    if args.models:
        targets = [t for t in TARGETS if t["label"] in set(args.models)]

    rows = []
    for t in targets:
        if t["provider"] not in providers:
            rows.append({**t, "status": f"skipped (no provider '{t['provider']}' in config)"})
            print(f"· {t['label']:<20} skipped — provider '{t['provider']}' not in {os.path.basename(args.config)}")
            continue
        print(f"▶ {t['label']:<20} ({t['model']} @ {t['provider']}) — {args.trials} trials …")
        try:
            llm = _build_llm(cfg, t)
        except Exception as exc:  # noqa: BLE001
            rows.append({**t, "status": f"build-failed: {type(exc).__name__}: {exc}"})
            print(f"  build failed: {exc}")
            continue

        raw_msgs = _build_history(raw_prompt)
        with_msgs = _build_history(prompt_with)
        raw_out = _run_condition(llm, raw_msgs, args.trials)
        with_out = _run_condition(llm, with_msgs, args.trials)

        errors = sum(1 for o in raw_out + with_out if o.startswith("<ERROR"))
        raw_leaks = sum(1 for o in raw_out if _has_marker(o))
        with_leaks = sum(1 for o in with_out if _has_marker(o))
        san_leaks = sum(1 for o in with_out if _has_marker(BaseAgent.sanitize_output(o)))
        rows.append({
            **t, "status": "ok", "errors": errors,
            "raw_leaks": raw_leaks, "with_prompt_leaks": with_leaks,
            "sanitized_leaks": san_leaks, "trials": args.trials,
            "samples": {"raw": raw_out, "with_prompt": with_out},
        })

    # ---- overview ----
    print("\n" + "=" * 78)
    print("LEAK-PROBE OVERVIEW  (lower = better; sanitized MUST be 0)")
    print("=" * 78)
    hdr = f"{'model':<20} {'chat':<5} {'raw':>8} {'+prompt':>9} {'sanitized':>10} {'err':>4}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['label']:<20} {'✓' if r.get('chat_chain') else '':<5} {r.get('status')}")
            continue
        n = r["trials"]
        chain = "✓" if r.get("chat_chain") else ""
        print(f"{r['label']:<20} {chain:<5} {r['raw_leaks']:>4}/{n:<3} "
              f"{r['with_prompt_leaks']:>4}/{n:<3} {r['sanitized_leaks']:>5}/{n:<4} {r['errors']:>4}")
    print("-" * 78)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nfull results → {args.json}")

    # Non-zero exit if the deterministic backstop ever failed.
    failed = [r for r in rows if r.get("status") == "ok" and r["sanitized_leaks"] > 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
