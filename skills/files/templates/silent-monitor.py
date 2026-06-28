#!/usr/bin/env python3
"""Silent-Monitor — generic skeleton for a cyclic monitor job.

The script decides for itself whether there is anything to report:
  * nothing to say  -> silent()  (empty stdout = no notification)
  * something to say -> emit(text) (the message becomes the notification)

Skill-Creator extends this with a deterministic data source and a gate.
Domain logic, vendor URLs, and credentials belong in the per-job params
(``--params '{"key": "value"}'``) — not in this skeleton.

Run directly during development:

    PAWLIA_SESSION_DIR=/path/to/session \\
    PAWLIA_USER_ID='<user-id>' \\
    python silent-monitor.py

Production (scheduled by ``automation add-job --script``) gets the same
env vars injected by the harness automatically.
"""
from pawlia.automation_harness import get_params, emit, silent, log

params = get_params()

# TODO(skill-creator): replace the body with a deterministic check.
#   1. Fetch from a stable source (HTTP API, local file, cached value).
#   2. Apply a gate — does the current state warrant a notification?
#   3. silent() on a quiet day, emit(formatted_message) when something is up.
#
# Example shape (DO NOT ship as-is — fill in the real source and gate):
#
#   data = fetch_from_source(params["source"])
#   if gate_passes(data):
#       silent()
#   else:
#       emit(format_message(data))

silent()  # default: quiet until the real logic is in place
