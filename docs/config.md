# Configuration

PawLia is configured via `config.yaml`. Copy `config.sample.yaml` as a starting point — it contains all available options with inline comments.

## Providers

Define one or more LLM backends. Any OpenAI-compatible API works.

```yaml
providers:
  ollama:
    apiBase: http://localhost:11434/v1
    apiKey: ollama        # required by some clients, value doesn't matter for Ollama
    timeout: 240          # seconds; increase for slow hardware
    keepAlive: -1         # keep model loaded indefinitely (-1 = forever)
  groq:
    apiBase: https://api.groq.com/openai/v1
    apiKey: gsk_...
```

| Key | Description |
|-----|-------------|
| `apiBase` | Base URL of the OpenAI-compatible API |
| `apiKey` | API key (required for cloud providers) |
| `timeout` | Request timeout in seconds |
| `keepAlive` | Ollama keep-alive duration (`-1` = forever, `0` = unload after each request) |

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
```

| Key | Description |
|-----|-------------|
| `model` | Model name as understood by the provider |
| `provider` | Key from `providers:` |
| `temperature` | Sampling temperature |
| `think` | Enable chain-of-thought / extended thinking (optional) |

## Agents

Assign model keys to agent types. Only configure what you want to override — the rest falls back automatically.

```yaml
agents:
  default: smart        # global fallback — required
  chat: smart           # main conversation agent
  skill_runner: fast    # default for all skill sub-agents
  vision: vision        # used when the user sends an image
  skills:               # per-skill overrides
    searxng: groq-fast
    browser: smart
```

### Fallback chain

| Agent type | Resolution order |
|------------|-----------------|
| `chat` | `agents.chat` → `agents.default` |
| `skill_runner` | `agents.skill_runner` → `agents.default` |
| `vision` | `agents.vision` → `agents.chat` → `agents.default` |
| `skill.<name>` | `agents.skills.<name>` → `agents.skill_runner` → `agents.default` |

LLMs with identical configuration are reused across agent types — no redundant connections.

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
  silence_threshold: 0.02
  silence_seconds: 1.5
  min_speech_seconds: 0.4
  min_active_speech_ratio: 0.12
  min_consecutive_speech_frames: 8
  call_inactivity_seconds: 180
```

| Key | Description |
|-----|-------------|
| `voip.silence_threshold` | Per-frame RMS threshold above which incoming audio counts as speech |
| `voip.silence_seconds` | Silence duration that closes the current VoIP speech chunk |
| `voip.min_speech_seconds` | Minimum chunk duration before deeper speech/noise analysis runs |
| `voip.min_active_speech_ratio` | Minimum share of active 20 ms frames required before a chunk is sent to STT |
| `voip.min_consecutive_speech_frames` | Minimum sustained run of active 20 ms frames required before a chunk is sent to STT |
| `voip.call_inactivity_seconds` | Hang up the VoIP call when no speech chunk has been sent to STT for this many seconds |

  webhook:
    port: 8080
    # token: OPTIONAL_BEARER_TOKEN       # enables Bearer auth on /chat
```

## Transcription (Speech-to-Text)

Used for voice messages in Telegram and Matrix, and for VoIP calls.

```yaml
transcription:
  provider: groq          # groq | openai | local

  groq:
    api_key: YOUR_GROQ_API_KEY
    model: whisper-large-v3-turbo
    # language: de

  # openai:
  #   api_key: YOUR_API_KEY
  #   base_url: https://api.openai.com/v1
  #   model: whisper-1

  # local:                              # no API key; requires FFmpeg + faster-whisper
  #   model: base                       # tiny | base | small | medium | large-v3
  #   device: cpu                       # cpu | cuda
  #   compute_type: int8
```

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

## Skill Configuration

Per-skill settings (URLs, API keys, etc.). Keys match the skill name.

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
| `markdown` | **yes** | LLM extracts topics from conversations and writes one Markdown file per topic. Retrieval via keyword matching — no embeddings, no extra dependencies. |
| `lightrag` | | Knowledge-graph RAG (powerful, slow). Requires `lightrag-hku`. |
| `simple` | | Chunk + embed + cosine similarity. Fast, numpy only. |
| `mem0` | | Fact extraction via mem0. Requires `mem0ai` + `chromadb`. |

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
