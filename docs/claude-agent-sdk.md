# Claude Agent SDK (Python)

Kelet observability for [`claude-agent-sdk`](https://docs.anthropic.com/en/api/agent-sdk/overview)
— captures every `query()` / `ClaudeSDKClient` call, preserves extended thinking
(which Claude Code otherwise redacts), and lets Claude Code's native OTLP
pipeline flow straight into your Kelet session.

## Version floor

- **Claude Code CLI**: v2.1.119 or later (`claude --version`).
- **claude-agent-sdk**: `>= 0.1.45`.

Older CLIs don't emit OTLP natively; older SDKs miss the stream hooks the
observer needs.

## Install

```bash
uv add 'kelet[claude-agent-sdk]'
# or
pip install 'kelet[claude-agent-sdk]'
```

This pulls in `claude-agent-sdk>=0.1.45` alongside Kelet. You still need the
Claude Code CLI on `PATH` (or pointed at via the SDK's
`path_to_claude_code_executable` option).

## Configure

```python
import kelet

kelet.configure(api_key="...", project="production")
```

`configure()` auto-detects `claude-agent-sdk` and installs the reasoning
observer — there is nothing else to import or wire.

Environment variables work equivalently:

```bash
export KELET_API_KEY=...
export KELET_PROJECT=production
```

## Two layers of telemetry

1. **Claude Code's native OTLP** — emits `claude_code.interaction`,
   `claude_code.llm_request`, `claude_code.tool` spans + log events for
   hooks, skills, compaction, permission-mode changes, etc. You enable
   this by setting the OTEL env vars before your process starts:

   ```bash
   export CLAUDE_CODE_ENABLE_TELEMETRY=1
   export OTEL_LOGS_EXPORTER=otlp
   export OTEL_METRICS_EXPORTER=otlp
   export OTEL_TRACES_EXPORTER=otlp
   export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
   export OTEL_EXPORTER_OTLP_ENDPOINT=$KELET_API_URL  # default: https://api.kelet.ai
   export OTEL_EXPORTER_OTLP_HEADERS="authorization=$KELET_API_KEY,x-kelet-project=$KELET_PROJECT"
   ```

   Kelet's SDK does NOT set these — the host app owns its subprocess env.

2. **Kelet reasoning observer** — installed by `configure()`. Captures
   `ThinkingBlock` text from the SDK's in-process message stream (Claude
   Code redacts reasoning in its OTLP) and emits it as a parallel
   `kelet.reasoning` OTLP log record. Nothing for you to call — the observer
   runs transparently when `configure()` sees `claude_agent_sdk` installed.

## What flows into Kelet

| Source | What it becomes |
|---|---|
| `claude_code.interaction` span | Session envelope |
| `claude_code.llm_request` span | `COMPLETION` run |
| `claude_code.tool` span | `TOOL` sub-run folded into the owning completion |
| `hook_execution_start` + `_complete` log events | `HOOK` run |
| `skill_activated` log event | `SKILL` run |
| `compaction` log event | `COMPACTION` run |
| `permission_mode_changed` log event | `PERMISSION_MODE_CHANGE` run |
| `kelet.reasoning` log event | Thinking text attached to the owning completion |

## See also

- [Contract spec](./claude-agent-sdk-contract.md) — exact attribute keys + invariants.
