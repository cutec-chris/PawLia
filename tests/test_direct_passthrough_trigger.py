"""Tests for the relaxed skill-creator direct-passthrough trigger.

The original pattern only fired on English imperatives starting with
``fix/improve/... <name>`` and missed every German phrasing the
dispatcher produces ("Der X Skill gibt Y. Bitte behebe das."). This
file pins down the cases that must keep working after the rewrite so
we do not regress when someone reorders the regex.
"""

import pytest

from pawlia.agents.skill_runner import SkillRunnerAgent


# Extract the static pieces we need without instantiating the agent
# (its constructor expects a real LLM + skill).
_VERBS = SkillRunnerAgent._DIRECT_CODE_VERBS
_BLOCKERS = SkillRunnerAgent._DIRECT_CODE_BLOCKERS


def _parse(q: str):
    """Replicate ``_try_direct_coding_backend``'s parse stage."""
    q_lower = q.lower().strip()
    if any(b in q_lower for b in _BLOCKERS):
        return None
    import re

    verb_found, verb_match = None, None
    for verb in _VERBS:
        m = re.search(rf"\b{re.escape(verb)}\b", q_lower)
        if m:
            verb_found, verb_match = verb, m
            break
    if not verb_found:
        return None
    return verb_found


# ── German verbs are recognised ────────────────────────────────────────


@pytest.mark.parametrize("verb", [
    "implementiere", "repariere", "behebe", "verbessere",
    "aktualisiere", "erstelle", "baue", "erweitere",
    "korrigiere", "debugge", "modernisiere",
])
def test_german_verbs_in_table(verb):
    assert verb in _VERBS


# ── Verb may appear anywhere in the query, not just at the start ────────


def test_verb_after_skill_name_with_colon():
    v = _parse("thunderstorm-alert fix: die Alarme sind falsch.")
    assert v == "fix"


def test_verb_at_end_after_description():
    v = _parse(
        "Der thunderstorm-alert Skill gibt falsche Gewitter-Alarme aus. "
        "Bitte behebe das."
    )
    assert v == "behebe"


def test_german_imperative_with_article():
    v = _parse("Bitte repariere den thunderstorm-alert skill.")
    assert v == "repariere"


def test_verb_inside_long_sentence():
    v = _parse(
        "das ist doch unsinn, der skill-creator soll den gewitter skill "
        "nochmal gründlich fixen."
    )
    assert v == "fixen" if "fixen" in _VERBS else None
    # "fixen" is the colloquial infinitive; if it is not registered we
    # still expect either a recognised verb in the same sentence.
    if v != "fixen":
        v2 = _parse(
            "das ist doch unsinn, der skill-creator soll den gewitter skill "
            "nochmal gründlich reparieren."
        )
        assert v2 in ("repariere", "verbessere", "implementiere")


# ── Things that must NOT trigger ────────────────────────────────────────


@pytest.mark.parametrize("q", [
    "change the coding backend to aider",       # config blocker
    "validate the skill-thunderstorm-alert",   # verb missing
    "was ist X?",                              # question, no verb
    "show current configuration",              # blocker phrase
])
def test_does_not_trigger(q):
    assert _parse(q) is None


# ── Blocker phrases ────────────────────────────────────────────────────


def test_blocker_set_contains_expected_phrases():
    for phrase in ("coding backend", "sync --workspace", "show current"):
        assert phrase in _BLOCKERS
