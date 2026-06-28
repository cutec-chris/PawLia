# Plan: Automation-Skill — stille Monitor-Jobs

## Context

Zyklische Monitor-Jobs (Gewitter-Check, Bahn-Check) spammen, weil der Automation-Skill
keine echten Script-Jobs erstellen kann. Script-Jobs können still bleiben (leere stdout =
kein Notify); Instruction-Jobs können es nicht — der LLM produziert immer Text.

Das Ziel: wenn jemand "fix das" schreibt, soll Thalia den Gewitter-Monitor (oder jeden
anderen kaputten Monitor) selbst reparieren können — ohne manuelle SSH-Intervention.

## Designprinzip: System über Lösungen

Vor dem ersten Fix: kein "bau ein DWD-Gewitter-Script ins Core". Einzelfix-Lösungen in
`pawlia/` oder `skills/files/templates/` (vendor-spezifische Templates) sind der
falsche Hebel — sie koppeln das System an einen Use-Case. Stattdessen: System reparieren
(Pfad-Auflösung, Skill-Scaffolding, Workflow-Building-Blocks, Doku), Domain-Scripts
on-demand von skill-creator bauen lassen. Vendor-spezifische Templates
(`gewitter-monitor.py` etc.) fliegen raus, generische Skeletons (z.B. `silent-monitor.py`)
sind okay.

→ ausführlich: `vision.md` › System over solutions und `agents.md` › Development
Guidelines.

## Root Causes

### RC1: `workflow.yaml` kennt nur `--instruction`
`skills/automation/workflow.yaml` — der `add-job`-Block erzwingt Instruction-Jobs.
Das `--script`-Flag existiert in `organizer.py` seit langem, ist aber nie im Workflow
gelandet. Kein Agent kann per normalem Skill-Aufruf einen Script-Job anlegen.

### RC2: Kein klarer Zwei-Schritt-Fluss dokumentiert
Wenn ein Agent einen Monitor bauen soll:
1. Skill-Creator schreibt das Script → `workspace/skills/scripts/<name>.py`
2. Automation registriert den Job mit `--script <name>.py`

Schritt 1 war in skill-creator SKILL.md (L131–160) beschrieben — aber mit falschem Pfad
(`workspace/.scripts/`, nicht `workspace/skills/scripts/`).
Schritt 2 war nicht möglich (RC1) und der Ablauf zwischen den Skills nicht explizit.
→ Agenten griffen auf Instruction-Jobs zurück oder erstellten ein Script das nie
  referenziert wurde.

### RC3: Vendor-spezifisches Template im Core
`skills/files/templates/gewitter-monitor.py` rief einen `thunderstorm-alert` Workspace-Skill
auf — der nirgends existiert und nicht im Bundled-Set ist. Wenn Skill-Creator das
Template als Basis nahm, baute er ein Script das beim ersten Run sofort still scheiterte
(`SCRIPT.is_file()` → False → `{}` → `silent()`), ohne Fehlermeldung. Der Job lief dann
scheinbar stumm, machte aber nichts. → Template gelöscht, ersetzt durch generisches
`silent-monitor.py`-Skeleton.

## Änderungen

### 1. `pawlia/utils.py` — `resolve_script()` erweitern

Primären Pfad `workspace/skills/scripts/` ergänzen, alte als Legacy-Fallback behalten.
Fehlendes Script gibt den primären Pfad zurück (Caller sieht "not found" statt falsches
Verzeichnis).

### 2. `pawlia/automation.py` — `_is_allowed_path()` erweitern

`workspace/skills/scripts/` als primären erlaubten Pfad aufnehmen.

### 3. `skills/automation/workflow.yaml` — `add-script-job` Building Block

Neuer Building Block `add-script-job` (mit `--script` + `--params`). `add-job`
Description: "for one-shot tasks that always deliver output; for silent monitors use
add-script-job".

### 4. `skills/automation/SKILL.md` — Monitor vs. One-Shot explizit machen

Entscheidungstabelle am Anfang von "Building a job", Pfad auf `workspace/skills/scripts/`
korrigiert, Hinweis: Skill-Creator baut das Script on-demand — kein Vendor-Template
im Core nötig.

### 5. `skills/skill-creator/SKILL.md` — Filesystem-Regeln schärfen

- Pfad L158: `workspace/.scripts/` → `workspace/skills/scripts/`.
- Tabelle L162+: skill-creator schreibt **ausschließlich** nach `workspace/skills/`
  (kein loses `workspace/`, kein `/tmp`).

### 6. `skills/files/templates/` — generisches Skeleton statt Vendor-Template

- `gewitter-monitor.py` gelöscht.
- `silent-monitor.py` neu: nur Harness-Skelett, kein Vendor-Coupling, mit
  `TODO(skill-creator)`-Kommentarblock.

### 7. `skills/organizer/scripts/organizer.py` — Help-Text

`--script` Help-Text zeigt den neuen primären Pfad und die Legacy-Fallbacks.

### 8. `vision.md` — neue Design-Decision

"System over solutions" unter Design Decisions eingefügt.

### 9. `agents.md` — neue Dev-Guideline

"System fixes, not one-off solutions" unter Development Guidelines, mit Verweis auf
die Vision-Decision.

## Was das NICHT ändert

- `organizer.py` CLI-Funktionalität (`--script` ist bereits implementiert)
- `pawlia/automation.py` SILENT_SENTINEL-System
- skill-creator L131-160 inhaltlich (nur Pfad-Korrektur + FS-Tabelle)

## Verifikation

End-to-End auf central: **"bau mir einen Gewitter-Monitor für Biederitz"** — Thalia
produziert aus dem Harness-Skeleton (nicht aus einem Vendor-Template) ein eigenständiges
Script in `workspace/skills/scripts/gewitter_check.py`, registriert den Job via
`add-script-job`, manueller `run-job` zeigt: silent bei Ruhe, Meldung bei aktiver
Warnung. **Kein SSH, kein DWD-Pfad im Core.**

Tests via `.venv/bin/python -m pytest tests/ -q`.
