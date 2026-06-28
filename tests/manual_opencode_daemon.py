"""Manual smoke test for the opencode daemon.

This is **not** part of the regular pytest run — it talks to a real
``opencode serve`` process and is slow (model invocation latency).
Run it explicitly when you change ``pawlia/coding/opencode_daemon.py``:

    source .venv/bin/activate
    python tests/manual_opencode_daemon.py

Expected output: a single dict with ``ok: true``, a short
``output`` (the model's reply), and a non-empty ``session_id`` on
the second call. The second call's ``session_id`` must match the
first one's, proving that the daemon kept the conversation alive
across invocations.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pawlia.coding.opencode_daemon import run_task  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        print(f"== workspace: {tmp}", flush=True)

        r1 = run_task(
            tmp,
            "Antworte mit genau den drei Wörtern: daemon ist wach.",
            user_id="smoke-test",
        )
        print("== call 1:", json.dumps(r1, indent=2, ensure_ascii=False), flush=True)
        assert r1["ok"], r1
        assert "daemon" in r1["output"].lower(), r1
        assert "wach" in r1["output"].lower(), r1

        r2 = run_task(
            tmp,
            "Und jetzt sag hallo dazu.",
            user_id="smoke-test",
        )
        print("== call 2:", json.dumps(r2, indent=2, ensure_ascii=False), flush=True)
        assert r2["ok"], r2

        # Same session = conversation continues across calls.
        assert r1["session_id"] == r2["session_id"], (
            f"session changed: {r1['session_id']!r} -> {r2['session_id']!r}"
        )
        # And the model remembered: "hallo" must be in the answer.
        assert "hallo" in r2["output"].lower(), r2

    print("OK: daemon is alive, follows up, and reuses the session", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
