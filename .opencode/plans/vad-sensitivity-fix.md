# Fix: VAD-Sensitivity für leise Umgebungen

## Problem

In leisen Umgebungen werden Sprach-Chunks als "background noise" verworfen, obwohl der User spricht. Der User meldet: "Ich habe eigentlich gerade eine ganze Menge erzählt und das ist alles nicht angekommen."

### Root Cause Analysis

1. **AGC nur zeitlich begrenzt aktiv**: `_activate_agc()` wird nur nach Bot/User-Aktivität aufgerufen (15s Fenster). In leisen Pausen fällt Gain auf 1.0 zurück.

2. **Fixer `SILENCE_THRESHOLD = 0.018`**: Leise Sprache mit RMS 0.02-0.05 hat nur wenige Frames darüber.

3. **`MIN_ACTIVE_SPEECH_RATIO = 0.12`**: Chunks mit 4-10% active_ratio werden verworfen.

### Audio-Analyse (Debug WAVs)

| Call | RMS-Bereich | Active% | LongestRun | Ergebnis |
|------|-------------|---------|------------|----------|
| d0e1e43a (leise) | 0.014–0.134 | 2–44% | 2–33 | Viele verworfen |
| 829c9a76 (besser) | 0.044–0.124 | 14–54% | 18–41 | Meiste transkribiert |

### Log-Belege (skipped Chunks)

```
skipping chunk as background noise (active_ratio=0.10 longest_run=21 ...) – 14.8s!
skipping chunk as background noise (active_ratio=0.05 longest_run=2 ...)
skipping chunk as background noise (active_ratio=0.06 longest_run=3 ...)
skipping chunk as background noise (active_ratio=0.04 longest_run=2 ...)
skipping chunk as background noise (active_ratio=0.07 longest_run=4 ...)
skipping chunk as background noise (active_ratio=0.08 longest_run=3 ...)
skipping chunk as background noise (active_ratio=0.04 longest_run=2 ...)
skipping chunk as background noise (active_ratio=0.02 longest_run=1 ...)
skipping chunk as background noise (active_ratio=0.02 longest_run=2 ...)
```

---

## Änderungen

Alle Änderungen in `pawlia/interfaces/matrix_call.py`.

### 1. AGC_QUIET_TARGET_RMS Konstante hinzufügen

**Zeile ~383-387** – Neue Konstante für leise Umgebungen:

```python
# AGC: boost gain in windows where we expect the user to speak
AGC_WINDOW_SECONDS = 15.0     # how long the AGC stays active after bot/user activity
AGC_TARGET_RMS = 0.10         # target RMS level for normalization (when bot active)
AGC_QUIET_TARGET_RMS = 0.06   # target RMS when bot is idle (boosts quiet speech)
AGC_MAX_GAIN = 12.0           # don't amplify more than this
AGC_SMOOTHING = 0.15          # EMA alpha for gain updates (higher = faster)
```

### 2. `_agc_rms()` – Immer aktiv mit adaptivem Target

**Zeile ~484-501** – AGC immer aktiv, aber mit zwei Targets:

```python
def _agc_rms(self, raw_rms: float) -> float:
    """Return the AGC-adjusted RMS for VAD decisions.

    Always active — uses AGC_TARGET_RMS when the bot is speaking/generating
    (avoids amplifying over its own TTS) and AGC_QUIET_TARGET_RMS otherwise
    to boost quiet speech in silent environments.
    """
    if raw_rms > 1e-6:
        # Lower target when bot is idle → amplifies quiet speech
        target = self.AGC_TARGET_RMS if self._bot_is_active() else self.AGC_QUIET_TARGET_RMS
        ideal_gain = target / raw_rms
        ideal_gain = min(ideal_gain, self.AGC_MAX_GAIN)
        alpha = self.AGC_SMOOTHING
        self._agc_gain = alpha * ideal_gain + (1 - alpha) * self._agc_gain

    return raw_rms * self._agc_gain
```

**Wichtig:** Die `_agc_active` Property-Prüfung wird entfernt – AGC ist jetzt immer aktiv.

### 3. `_should_transcribe_chunk()` – Adaptive Schwellwerte

**Zeile ~1380-1401** – Niedrigere Schwellwerte bei leisem Noise Floor:

```python
def _should_transcribe_chunk(
    self,
    pcm: "np.ndarray",
    sample_rate: int,
    fps: int,
) -> bool:
    """Return True only when a chunk contains sustained speech-like activity."""
    stats = self._analyze_speech_chunk(pcm, sample_rate, fps)

    # In very quiet environments (low noise floor), relax the active_ratio
    # threshold. When background noise is near zero, any signal above the
    # silence threshold is likely speech, not noise.
    if self._noise_floor < 0.001:
        min_active_ratio = self.MIN_ACTIVE_SPEECH_RATIO * 0.5  # 0.06 statt 0.12
        min_speech_like = self.MIN_SPEECH_LIKE_RATIO * 0.5     # 0.04 statt 0.08
    else:
        min_active_ratio = self.MIN_ACTIVE_SPEECH_RATIO
        min_speech_like = self.MIN_SPEECH_LIKE_RATIO

    basic_match = (
        stats["active_ratio"] >= min_active_ratio
        and stats["longest_run"] >= self.MIN_CONSECUTIVE_SPEECH_FRAMES
        and stats["speech_like_ratio"] >= min_speech_like
        and stats["speech_like_run"] >= self.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES
    )
    if not basic_match:
        return False
    if self._webrtc_vad is None:
        return True
    return (
        stats["voiced_ratio"] >= self.WEBRTC_VAD_MIN_VOICED_RATIO
        and stats["voiced_run"] >= self.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES
    )
```

### 4. Speech-triggered AGC Activation

**Zeile ~1855-1860** (in `_audio_pipeline()`) – AGC sofort bei Spracherkennung:

```python
if speech_like_frame:
    if not speech_buffer and silence_count == 0:
        self._activate_agc()  # AGC window on speech detection
        logger.info("call %s: speech started (rms=%.4f)",
                    self.call_id[:8], rms)
        self._mark_user_speech_started()
        speech_buffer = self._start_speech_buffer(pre_speech_buffer, pcm)
    else:
        speech_buffer.append(pcm)
```

### 5. Config-Option `voip.agc_always_active` (optional)

**Zeile ~511-600** (in `_load_voip_audio_config()`) – Config für AGC-Verhalten:

```python
# Nach den existing voip config loads:
self.AGC_QUIET_TARGET_RMS = self._get_float_config(
    voip_cfg,
    "agc_quiet_target_rms",
    self.AGC_QUIET_TARGET_RMS,
    minimum=0.01,
    maximum=1.0,
)
```

**config.yaml Erweiterung:**

```yaml
voip:
  # AGC target when bot is idle (lower = more aggressive boost for quiet speech)
  agc_quiet_target_rms: 0.06
```

---

## Test-Plan

1. **Unit-Test**: `_agc_rms()` mit verschiedenen RMS-Werten und Bot-Status
2. **Integration**: `_should_transcribe_chunk()` mit simulierten leisen Chunks
3. **Manuell**: Call in leiser Umgebung testen – Chunks sollten nicht mehr verworfen werden

## Risiko-Bewertung

| Änderung | Risiko | Begründung |
|----------|--------|------------|
| AGC immer aktiv | Niedrig | `_bot_is_active()` schützt vor TTS-Feedback |
| Adaptive ratio | Niedrig | Nur bei noise_floor < 0.001 (sehr leise) |
| Speech-triggered AGC | Niedrig | `_activate_agc()` existiert bereits |
| Config-Option | Keins | Nur additive Erweiterung |

## Fallback

Wenn Probleme: `agc_quiet_target_rms: 1.0` in config.yaml setzt Target so hoch, dass kein Boost erfolgt (effektiv wie vorher).
