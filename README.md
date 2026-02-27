<div align="center">
  <img src=".github/logo.png" alt="Kelet" width="400">

  <h1>Automated Root Cause Analysis for AI Agents</h1>

  <p><strong>Agent failures take weeks to diagnose manually. Kelet runs 24/7 deep diagnosis and suggests targeted fixes.</strong></p>

  <img src=".github/illustration.png" alt="Kelet workflow" width="700">
</div>

Kelet analyzes production failures 24/7. Each trace takes 15-25 minutes to debug manually—finding patterns requires analyzing hundreds of traces. That's **weeks of engineering time** per root cause. Kelet does this automatically, surfacing issues like data imbalance, concept drift, prompt poisoning, and model laziness hidden in production noise.

---

## What Kelet Does

Kelet runs 24/7 analyzing every production trace:

1. **Captures** every interaction, user signal, and failure context automatically
2. **Analyzes** hundreds of failures in parallel to detect repeatable patterns
3. **Identifies** root causes (data issues, prompt problems, model behavior)
4. **Delivers** targeted fixes, not just dashboards

Unlike observability tools that show you data, Kelet analyzes it and tells you what to fix.

**Not magic**: Kelet is in alpha. Won't catch everything yet, needs your guidance sometimes. But it's already doing analysis that would take weeks manually.

Three lines of code to start.

## Installation

**Using uv (recommended):**
```bash
uv add git+https://${GITHUB_KELET_PAT}@github.com/Kelet-ai/python-sdk.git
```

**Or using pip:**
```bash
pip install git+https://${GITHUB_KELET_PAT}@github.com/Kelet-ai/python-sdk.git
```

Set your API key:
```bash
export KELET_API_KEY=your_api_key
export KELET_PROJECT=production  # Optional: organize traces by environment
```

Or configure in code:
```python
kelet.configure(
    api_key="your_api_key",
    project="production"  # Groups traces by project/environment
)
```

## Quick Start

```python
import kelet

kelet.configure()  # Auto-instruments pydantic-ai and captures sessions

# Your agent code works as-is - instrumentation is automatic
result = await agent.run("Book a flight to NYC")

# Optionally capture user feedback
await kelet.signal(
    kind=kelet.SignalKind.FEEDBACK,
    source=kelet.SignalSource.HUMAN,
    score=0.0,  # User unhappy? Kelet analyzes why.
)
```

**That's it.** Kelet now runs 24/7 analyzing every trace, clustering failure patterns, and identifying root causes—work that would take weeks manually.

### Manual Session Grouping (Optional)

If your framework doesn't support session tracking, or you want custom session IDs:

```python
with kelet.agentic_session(session_id="user-123-request-456"):
    result = await agent.run("Book a flight to NYC")
```

But most users don't need this—instrumentation captures sessions automatically from pydantic-ai and other supported frameworks.

### Easy Feedback UI for React

Building a React frontend? Use the [Kelet Feedback UI](https://github.com/kelet-ai/feedback-ui) component for instant implicit and explicit feedback collection.
See the [live demo](https://feedback-ui.kelet.ai/) and [documentation](https://github.com/kelet-ai/feedback-ui) for full integration guide.

### Works with Your Observability Stack

Already using Logfire or another OTEL provider? Kelet integrates seamlessly:

```python
import logfire
import kelet

logfire.configure()
logfire.instrument_pydantic_ai()

kelet.configure()  # Adds Kelet's processor to your existing setup
```

---

## What Gets Captured

Kelet is built on [OpenTelemetry](https://opentelemetry.io/) and supports multiple semantic conventions for AI/LLM observability:

| Semantic Convention | Supported Frameworks |
|---------------------|----------------------|
| [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | Pydantic AI, LiteLLM, Langfuse SDK |
| Vercel AI SDK | Next.js, Vercel AI |
| OpenInference | Arize Phoenix |
| OpenLLMetry / Traceloop | LangChain, LangGraph, LlamaIndex, OpenAI SDK, Anthropic SDK |

Any framework that exports OpenTelemetry traces using the GenAI semantic conventions will work automatically.

**Captured data includes:**

- **LLM calls**: Model, provider, tokens, latency, errors
- **Agent sessions**: Multi-step interactions grouped by user session
- **Custom context**: User IDs, session metadata, business-specific attributes

All captured automatically when you instrument with `kelet.configure()`.

---

## Configuration

Set via environment variables:

```bash
export KELET_API_KEY=your_api_key    # Required
export KELET_PROJECT=production      # Optional, defaults to "default"
export KELET_API_URL=https://...     # Optional, defaults to api.kelet.ai
```

Or pass directly to `configure()`:

```python
kelet.configure(
    api_key="your_api_key",
    project="production",
    auto_instrument=True  # Instruments pydantic-ai automatically
)
```

## API Reference

**Core Functions:**

```python
# Initialize SDK
kelet.configure(api_key=None, project=None, auto_instrument=True)

# Group operations by session for failure correlation
with kelet.agentic_session(session_id="session-id"):
    result = await agent.run(...)

# Capture user feedback
await kelet.signal(
    kind=kelet.SignalKind.FEEDBACK,       # feedback | edit | event | metric | arbitrary
    source=kelet.SignalSource.HUMAN,      # human | label | synthetic
    score=0.0,                            # 0.0 to 1.0
)

# Access current context
session_id = kelet.get_session_id()
trace_id = kelet.get_trace_id()
user_id = kelet.get_user_id()

# Manual shutdown (automatic on exit)
kelet.shutdown()
```

---

## Production-Ready

The SDK never disrupts your application:

- **Async**: Telemetry exports in background, zero blocking
- **Fail-safe**: Network errors handled silently, no exceptions raised
- **Graceful**: If Kelet is down, your agent keeps running
- **Auto-flush**: Spans exported automatically on process exit

---

## Alpha Status

Kelet is in alpha. What this means:

- **It works**: Already analyzing thousands of production traces for early users
- **Not perfect**: Won't catch every failure pattern yet, sometimes needs guidance
- **Improving fast**: The AI learns from more production data every day
- **We need feedback**: Help us make it better—tell us what it catches and what it misses

Even in alpha, Kelet does analysis that would take your team weeks to do manually.

**The alternative?** Manually analyzing 15-25 minutes per trace, across hundreds of failures, trying to spot patterns by hand. Most teams just don't do it—and ship broken agents.

---

## Learn More

- **Website**: [kelet.ai](https://kelet.ai)
- **Early Access**: We're onboarding teams with production AI agents
- **Support**: [GitHub Issues](https://github.com/Kelet-ai/python-sdk/issues)

Built for teams shipping mission-critical AI agents.
