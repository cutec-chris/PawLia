# Configuration

PawLia is configured via `config.yaml`. Copy `config.sample.yaml` as a starting point — it contains all available options with inline comments.

## Providers

Define one or more providers. A provider has a `backend` type plus any
backend-specific connection settings.

```yaml
providers:
  ollama:
    backend: pawlia
    apiBase: http://localhost:11434/v1
    apiKey: ollama        # required by some clients, value doesn't matter for Ollama
    timeout: 240          # seconds; increase for slow hardware
    keepAlive: -1         # keep model loaded indefinitely (-1 = forever)
  groq:
    backend: pawlia
    apiBase: https://api.groq.com/openai/v1
    apiKey: gsk_...
  hermes_local:
    backend: hermes
    apiBase: http://127.0.0.1:8642/v1
    apiKey: change-me
    timeout: 600
    conversation_namespace: pawlia
    store: true
```

| Key | Description |
|-----|-------------|
| `backend` | Backend type: `pawlia` (default if omitted) or `hermes` |
| `apiBase` | Base URL of the OpenAI-compatible API |
| `apiKey` | API key (required for cloud providers) |
| `timeout` | Request timeout in seconds |
| `keepAlive` | Ollama keep-alive duration (`-1` = forever, `0` = unload after each request) |
| `conversation_namespace` | Hermes only: prefix used for stable server-side conversation IDs |
| `store` | Hermes only: whether Hermes should persist response-chain state server-side |

### Backend behavior

#### `backend: pawlia`

This is the normal mode and also the default when `backend` is omitted.
PawLia uses its own stack:

- `LLMFactory` / `llm.py`
- `ChatAgent`
- `SkillRunnerAgent`
- bundled and workspace skills

#### `backend: hermes`

PawLia becomes a thin interface layer and forwards turns to Hermes via its
Responses API. Hermes keeps the live tool/runtime context. PawLia still:

- writes daily logs
- writes thread logs
- exposes the same interfaces (`Matrix`, `Telegram`, `Web`, `CLI`, `Webhook`)
- keeps Dream Wiki / summaries working from the visible chat transcript

In Hermes mode, PawLia does not use its own skill stack for that turn.

## Models

Named model definitions. Each bundles a model name, provider reference, and generation parameters. Agent types and skills reference models by key.

```yaml
models:
  fast:
    model: qwen3:4b
    provider: ollama
    temperature: 0.7
  smart:
    model: qwen3.5:latest
    provider: ollama
    temperature: 0.9
    think: true           # enable chain-of-thought (model must support it)
  vision:
    model: qwen2.5vl:latest
    provider: ollama
  groq-fast:
    model: qwen3:4b
    provider: groq
    temperature: 0.3
  hermes:
    model: hermes-agent
    provider: hermes_local
```

| Key | Description |
|-----|-------------|
| `model` | Model name as understood by the provider |
| `provider` | Key from `providers:` |
| `temperature` | Sampling temperature |
| `think` | Enable chain-of-thought / extended thinking (optional) |

The model itself does not choose the backend. The backend comes from the referenced provider.

## Agents

Assign model keys to agent types. Only configure what you want to override — the rest falls back automatically.

```yaml
agents:
  default: smart        # global fallback — required
  chat: smart,fast      # main conversation agent (runtime failover)
  skill_runner: fast    # default for all skill sub-agents
  vision: vision        # used when the user sends an image
  skills:               # per-skill overrides
    searxng: groq-fast,fast
    browser: smart,fast
```

If an agent value contains a comma-separated list, models are tried in order when an invocation fails (for example provider timeout or network/API errors).

### Fallback chain

| Agent type | Resolution order |
|------------|-----------------|
| `chat` | `agents.chat` → `agents.default` |
| `skill_runner` | `agents.skill_runner` → `agents.default` |
| `vision` | `agents.vision` → `agents.chat` → `agents.default` |
| `skill.<name>` | `agents.skills.<name>` → `agents.skill_runner` → `agents.default` |

LLMs with identical configuration are reused across agent types — no redundant connections.

If an agent resolves to a Hermes-backed model, PawLia routes that conversation through Hermes instead of its own chat/skill stack.

## Interfaces

Enable the interfaces you want to use. All enabled interfaces run simultaneously in server mode.

```yaml
interfaces:
  telegram:
    token: YOUR_BOT_TOKEN

  matrix:
    homeserver: https://matrix.org
    user_id: "@yourbot:matrix.org"
    password: YOUR_PASSWORD
    # access_token: OR_USE_THIS_INSTEAD_OF_PASSWORD
    # always_thread: true                  # always reply in a new thread (default: false)
    # stun_servers:
    #   - stun:stun.l.google.com:19302   # for VoIP calls

| Key | Description |
|-----|-------------|
| `always_thread` | When `true`, every message is answered in its own Matrix thread (default: `false`) |
| `stun_servers` | STUN/TURN server URIs for VoIP calls |

## VoIP

Shared VoIP behavior is configured globally so the same settings can be reused by Matrix today and other call transports later.

```yaml
voip:
  silence_threshold: 0.018
  silence_seconds: 1.5  # default
  min_speech_seconds: 0.4
  min_active_speech_ratio: 0.12
  min_consecutive_speech_frames: 8
  min_speech_band_ratio: 0.35
  max_spectral_flatness: 0.72
  min_speech_like_ratio: 0.08
  min_consecutive_speechlike_frames: 4
  webrtcvad_enabled: true
  webrtcvad_mode: 2
  webrtcvad_min_voiced_ratio: 0.12
  webrtcvad_min_consecutive_frames: 4
  call_inactivity_seconds: 180
  response_delay_seconds: 1.2
  agc_window_seconds: 15.0
  agc_target_rms: 0.10
  agc_max_gain: 12.0
  agc_smoothing: 0.15
  preanswer_warmup_enabled: true
  preanswer_warmup_timeout_seconds: 25.0
  preanswer_stt_silence_seconds: 0.4
```

| Key | Description |
|-----|-------------|
| `voip.silence_threshold` | Baseline per-frame RMS threshold for silence detection. Acts as a floor: the effective threshold is raised automatically when background noise is present (see adaptive silence below) |
| `voip.silence_seconds` | Silence duration that closes the current VoIP speech chunk |
| `voip.min_speech_seconds` | Minimum chunk duration before deeper speech/noise analysis runs |
| `voip.min_active_speech_ratio` | Minimum share of active 20 ms frames required before a chunk is sent to STT |
| `voip.min_consecutive_speech_frames` | Minimum sustained run of active 20 ms frames required before a chunk is sent to STT |
| `voip.min_speech_band_ratio` | Minimum share of frame energy that must lie in the speech band (roughly 180 Hz to 4 kHz) |
| `voip.max_spectral_flatness` | Upper limit for how noise-like active frames may be before they are rejected as non-speech |
| `voip.min_speech_like_ratio` | Minimum share of frames that must simultaneously be active and speech-like before a chunk is sent to STT |
| `voip.min_consecutive_speechlike_frames` | Minimum sustained run of speech-like frames required before a chunk is sent to STT |
| `voip.webrtcvad_enabled` | Enable an additional lightweight WebRTC speech detector before sending audio to STT |
| `voip.webrtcvad_mode` | WebRTC VAD aggressiveness from `0` (lenient) to `3` (strict) |
| `voip.webrtcvad_min_voiced_ratio` | Minimum share of frames WebRTC VAD must classify as voiced before a chunk is sent to STT |
| `voip.webrtcvad_min_consecutive_frames` | Minimum sustained run of WebRTC-voiced frames required before a chunk is sent to STT |
| `voip.call_inactivity_seconds` | Hang up the VoIP call when no speech chunk has been sent to STT for this many seconds |
| `voip.response_delay_seconds` | Minimum quiet time after the latest caller speech before PawLia starts answering. The actual delay scales up automatically with long monologues (see adaptive response delay below) |
| `voip.agc_window_seconds` | How long PawLia keeps automatic gain control active after recent speech / call activity |
| `voip.agc_target_rms` | Target loudness AGC tries to normalize incoming audio toward for VAD decisions |
| `voip.agc_max_gain` | Upper amplification cap AGC may apply to quiet incoming audio |
| `voip.agc_smoothing` | How quickly AGC adapts toward the target loudness (`1.0` = very fast) |
| `voip.preanswer_warmup_enabled` | Warm STT with silent audio and prepare the LLM/TTS greeting before sending `m.call.answer` |
| `voip.preanswer_warmup_timeout_seconds` | Maximum time to wait for pre-answer warmup before answering anyway |
| `voip.preanswer_stt_silence_seconds` | Duration of the silent WAV sent through STT during pre-answer warmup |

### Adaptive silence detection

The pipeline tracks a rolling EMA of the background noise floor (measured during inter-speech periods). The frame-level silence gate then raises the effective threshold to `max(silence_threshold, noise_floor × 2)` so that steady background noise — road noise, cycling, wind — falls below the effective threshold and counts as silence. This prevents speech chunks from growing indefinitely when the raw RMS never drops all the way to zero.

`silence_threshold` remains the configurable floor; the adaptive part only ever raises it, never lowers it.

### Adaptive response delay

After a speech chunk is accepted the pipeline waits at least `response_delay_seconds` before generating a reply. The actual delay scales with how long the caller just spoke:

| Last speech duration | Minimum wait |
|----------------------|--------------|
| < 6 s | `response_delay_seconds` (default 1.2 s) |
| 6 – 12 s | 3.0 s |
| 12 – 20 s | 4.0 s |
| > 20 s | 5.0 s |

A small bonus (up to 1.5 s) is added when the measured noise floor is significantly elevated, because background noise can mask the true end of speech.

  webhook:
    port: 8080
    # token: OPTIONAL_BEARER_TOKEN       # enables Bearer auth on /chat
```

## Transcription (Speech-to-Text)

Used for voice messages in Telegram and Matrix, and for VoIP calls.

```yaml
transcription:
  # Explicit STT fallback list, tried top to bottom.
  # PawLia only uses providers listed here.
  providers:
    - name: lan-whisper
      provider: local
      base_url: http://192.168.177.120:8005/v1
      model: deepdml/faster-whisper-large-v3-turbo-ct2
      language: de
      timeout: 10
    - groq

  groq:
    api_key: YOUR_GROQ_API_KEY
    model: whisper-large-v3-turbo
    language: de
    timeout: 30

  preprocess:
    highpass_hz: 140
    lowpass_hz: 7000
    denoise_strength: 1.25
    denoise_floor: 0.2
    adaptive_gate_percentile: 0.2
    adaptive_gate_multiplier: 2.2
    gate_threshold: 0.015
    gate_ratio: 0.2

  # Provider configs are only used when referenced from `providers`.
  # openai:
  #   api_key: YOUR_API_KEY
  #   base_url: https://api.openai.com/v1
  #   model: whisper-1

  # local:
  #   # Self-hosted OpenAI-compatible Whisper endpoint:
  #   # base_url: http://127.0.0.1:8000/v1
  #   # model: whisper-large-v3-turbo
  #
  #   # Or in-process faster-whisper (no base_url; requires FFmpeg + faster-whisper):
  #   model: base                       # tiny | base | small | medium | large-v3
  #   device: cpu                       # cpu | cuda
  #   compute_type: int8
```

`transcription.providers` is the explicit STT fallback list. Use one entry for
no fallback, or multiple entries for fallback. PawLia does not try every
configured provider automatically. Each entry can be a provider name, or an
inline provider config. Inline configs are useful for per-provider `timeout`
values or for trying multiple endpoints of the same provider type. Runtime
failures are tracked across requests: after 3 failures a provider is skipped for
30 minutes, then tried again.

| Key | Description |
|-----|-------------|
| `transcription.providers` | Ordered STT fallback chain; tries entries until one returns text; providers are temporarily skipped after repeated runtime failures |
| `transcription.<provider>.timeout` | HTTP timeout in seconds for OpenAI-compatible transcription endpoints |
| `transcription.preprocess.highpass_hz` | Removes low-frequency rumble such as wind, handling noise or desk vibrations |
| `transcription.preprocess.lowpass_hz` | Cuts very high frequencies that mostly contain hiss and sharp background noise |
| `transcription.preprocess.denoise_strength` | Strength of spectral background-noise subtraction |
| `transcription.preprocess.denoise_floor` | Residual floor kept during denoising to avoid metallic artifacts |
| `transcription.preprocess.adaptive_gate_percentile` | Percentile used to estimate the chunk's noise floor for adaptive gating |
| `transcription.preprocess.adaptive_gate_multiplier` | How far above the estimated noise floor audio must rise before it is treated as likely speech |
| `transcription.preprocess.gate_threshold` | Minimum absolute level for the final soft gate |
| `transcription.preprocess.gate_ratio` | How much low-level residual audio is retained below the gate threshold |

## Text-to-Speech (VoIP)

Used to speak responses during Matrix VoIP calls. Responses are streamed sentence-by-sentence for lower latency.

```yaml
tts:
  provider: piper         # piper | edge

  piper:                  # local, no internet required
    executable: piper
    model: /app/piper/de_DE-kerstin-low.onnx
    config: /app/piper/de_DE-kerstin-low.onnx.json
    sample_rate: 16000

  # edge:                 # Microsoft Edge TTS (requires internet)
  #   voice: de-DE-KatjaNeural

  # hold_audio: /app/assets/keyboard.m4a   # background sound while waiting for agent (default: assets/keyboard.m4a)
```

| Key | Description |
|-----|-------------|
| `provider` | `piper` (local) or `edge` (Microsoft Edge TTS, requires internet) |
| `hold_audio` | Path to audio file (wav/mp3/m4a) played to caller while waiting. Default: `assets/keyboard.m4a` |

## Workspace

The workspace directory (`session/<user>/workspace/`) serves as an Obsidian vault. Optional Git integration auto-commits changes and keeps the repo compact with daily/weekly squash.

```yaml
workspace:
  git:
    enabled: true                  # auto-commit workspace changes
    daily_squash_time: "23:00"     # squash daily commits into one
    weekly_squash_day: 6           # 0=Mon..6=Sun (default: Sunday)
    weekly_squash_time: "23:30"    # squash weekly commits into one
    push: false                    # push to remote after squash
```

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable workspace Git auto-commit and squash |
| `daily_squash_time` | `23:00` | Time to squash all daily commits into one |
| `weekly_squash_day` | `6` | Day of week for weekly squash (0=Mon, 6=Sun) |
| `weekly_squash_time` | `23:30` | Time for weekly squash |
| `push` | `false` | Push to remote after squash (`--force-with-lease`) |

Auto-commits are throttled to max 1 per 5 minutes. See [automation.md](automation.md#workspace-git-sync) for details.

## Skill Configuration

Per-skill deployment settings (URLs, hosts, timeouts, model names, etc.). Keys
match the skill name. Secrets should use skill credentials instead.

```yaml
skill-config:
  searxng:
    url: http://localhost:8888
    timeout: 10
  perplexica:
    url: http://localhost:3000
```

### RAG backends (memory & researcher)

The `memory` and `researcher` skills index documents for later retrieval. The backend is selected via `rag_backend`:

```yaml
skill-config:
  memory:
    embedding_provider: ollama
    embedding_model: bge-m3:latest
    embedding_dim: 1024
    embedding_host: http://localhost:11434
    rag_backend: markdown          # markdown | lightrag | simple | mem0
    rag_model: qwen3.5:latest      # LLM for topic extraction / RAG
```

| Backend | Default | Description |
|---------|---------|-------------|
| `markdown` | **yes** | **Dream Wiki** — LLM builds a structured, interlinked wiki from conversations. Pages have YAML frontmatter, `[[wikilinks]]`, and cross-references. Runs automatically overnight when idle. No embeddings required. |
| `lightrag` | | Knowledge-graph RAG (powerful, slow). Requires `lightrag-hku`. |
| `simple` | | Chunk + embed + cosine similarity. Fast, numpy only. |
| `mem0` | | Fact extraction via mem0. Requires `mem0ai` + `chromadb`. |

### Dream Wiki (default `markdown` backend)

The default memory backend implements Karpathy's [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern. When the user has been idle for 20 minutes (typically overnight), the scheduler processes new daily chat logs and:

1. **Analyzes** each conversation log via LLM — extracts topics, entities, decisions, facts
2. **Creates/updates** wiki pages with structured Markdown, YAML frontmatter, and `[[wikilinks]]`
3. **Rebuilds** `index.md` (catalog of all pages) and `log.md` (chronological audit log)
4. **Consolidates** — merges overlapping pages, adds missing cross-references

Storage per user:
```
workspace/wiki/                   # In the Obsidian vault
  index.md                        # Catalog of all pages
  log.md                          # Chronological audit log
  topics/
    projekt-thalia.md             # One wiki page per topic/entity
    linux-admin.md

memory_index/                     # Outside the vault (internal tracking)
  dreamed_files.json              # Which logs have been processed
```

Manual commands via the memory skill:
- `dream` — trigger wiki consolidation immediately
- `lint` — health check: merge overlapping pages, fix missing links

| Key | Used by | Description |
|-----|---------|-------------|
| `embedding_provider` | lightrag, simple | `ollama` or OpenAI-compatible |
| `embedding_model` | lightrag, simple | Embedding model name |
| `embedding_dim` | lightrag, simple | Embedding dimensions |
| `embedding_host` | all | Ollama / API base URL |
| `rag_backend` | all | Backend selection (default: `markdown`) |
| `rag_provider` | markdown, lightrag, mem0 | LLM provider (defaults to `embedding_provider`) |
| `rag_model` | markdown, lightrag, mem0 | LLM model for indexing / queries |
| `rag_numctx` | markdown, lightrag | LLM context window (default: 4096) |
| `rag_timeout` | all | LLM timeout in seconds (default: 600) |

## Skill Installation

```yaml
skill-install:
  allow_remote: false     # allow skill upload via Telegram/Matrix file message
```
