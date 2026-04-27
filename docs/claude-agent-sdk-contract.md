# Claude Agent SDK Contract (Kelet)

Shared contract between Kelet's Python and TypeScript SDKs for observing the
Claude Agent SDK. Both SDKs satisfy this spec; conformance tests replay
captured OTLP fixtures against each.

## Why a contract?

Claude Code v2.1.119+ natively emits OTLP traces, logs, and metrics when fed
the right env vars. Kelet's SDK work is **minimal**: observe reasoning text
(which Claude Code redacts in its own OTLP) and publish it as a parallel
OTLP log record. Everything else — turn boundaries, tool invocations,
hooks, skills, compaction, permission-mode changes — comes straight from
Claude Code's native pipeline.

Having a shared contract means the extraction in
`workflows/src/otel/extraction.py` is language-agnostic — the same captured
OTLP from either SDK produces identical `session_log` XML.

## 1. Env-var recipe (caller-provided)

The SDK no longer injects OTLP env vars into the Claude subprocess. The host
app must set them before calling `query()` / `ClaudeSDKClient`:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.kelet.ai
export OTEL_EXPORTER_OTLP_HEADERS="authorization=$KELET_API_KEY,x-kelet-project=$KELET_PROJECT"
```

Or in-process (Python / TypeScript equivalents exist in the fixture
generators at `e2e/fixtures/claude-agent-sdk-{python,typescript}/`).

Kelet's `configure()` does NOT set these — callers own the subprocess
environment.

## 2. Reasoning observer (SDK-provided)

Claude Code redacts `thinking` text in its `api_response_body` log event
(`"thinking":"<REDACTED>"`). To preserve reasoning fidelity, both SDKs
observe the in-process message stream and emit a parallel OTLP log record
per `ThinkingBlock`.

### Event

| Field | Value |
|---|---|
| body | `kelet.reasoning` |
| `event.name` attribute | `kelet.reasoning` |
| emitted via | OTLP log pipeline (LoggerProvider) |
| endpoint | `{KELET_API_URL}/api/logs` |

### Required attributes

| Attribute | Type | Notes |
|---|---|---|
| `reasoning.text` | string | Full ThinkingBlock content (unredacted) |
| `reasoning.signature` | string | Opaque signature returned by the model; empty string when absent |

### Optional attributes

| Attribute | When present |
|---|---|
| `reasoning.message_id` | Present iff the `AssistantMessage` carries `message_id` (Py) or `message.id` (TS). Used by extraction to correlate back to the `claude_code.llm_request` response |
| `session.id` | Present iff the message carries a session id on the envelope |

### Emission points

Both SDKs wrap three async-iterator entry points:

- `claude_agent_sdk.query` (module-level factory)
- `ClaudeSDKClient.receive_messages` (instance method)
- `ClaudeSDKClient.receive_response` (convenience wrapper)

For each yielded `AssistantMessage`, scan `content[]` for entries with a
string `thinking` attribute and emit exactly one log record per block.

The SDK duck-types message shape: both `msg.content[]` (Py shape) and
`msg.message.content[]` (TS SDK v0.1.x shape) are accepted.

## 3. Extraction (consumer side)

`workflows/src/otel/extraction.py::_extract_claude_code` treats the
`kelet.reasoning` log record as a first-class source of reasoning text:

1. Log records are attached to the owning `claude_code.interaction` span by
   the merge activity (time-window correlation).
2. `_collect_kelet_reasoning_events` harvests them keyed by
   `reasoning.message_id`.
3. During per-`llm_request` extraction, the matching reasoning text is
   prepended as a `<thinking>...</thinking>` block in the completion output.

Back-compat: the v2 span-event form (name=`kelet.reasoning` on
`claude_code.sdk_query`) is still accepted so pre-recapture fixtures keep
working.

## 4. Version floor

| Component | Version |
|---|---|
| Claude Code CLI | ≥ 2.1.119 (native OTLP support) |
| `claude-agent-sdk` (Python) | ≥ 0.1.45 |
| `@anthropic-ai/claude-agent-sdk` (TypeScript) | ≥ 0.1.77 |

## 5. What the SDKs no longer do

v3 removed:

- **Env-var injection** into `options.env` or `os.environ` — callers set
  them themselves per §1.
- **Parent `claude_code.sdk_query` span** — Claude Code's own
  `claude_code.interaction` is sufficient as the interaction root; no Kelet
  synthetic span wraps it.
- **`TRACEPARENT` serialization** — handled natively by Claude Code when the
  env vars are set.
- **ContextManager probe** (TS) — no parent span to propagate.
- **Shielded cleanup** — no span to end on exit paths.

The `installReasoningObserver()` (TS) and `ClaudeAgentSDKInstrumentor`
(Python) surfaces remain, but install the reasoning observer only.
