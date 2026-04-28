# CHANGELOG

<!-- version list -->

## v1.5.0 (2026-04-28)

### Bug Fixes

- **kel-391**: Address PR #9 review blockers (scope name, provider, sticky session)
  ([#9](https://github.com/Kelet-ai/python-sdk/pull/9),
  [`6e5fcb3`](https://github.com/Kelet-ai/python-sdk/commit/6e5fcb3df92e0b813faae8093fd3dd85a02e0d0c))

### Features

- **claude_agent_sdk**: Slim reasoning observer + auto-instrumentor
  ([#9](https://github.com/Kelet-ai/python-sdk/pull/9),
  [`6e5fcb3`](https://github.com/Kelet-ai/python-sdk/commit/6e5fcb3df92e0b813faae8093fd3dd85a02e0d0c))

- **configure**: Install LoggerProvider for OTLP log export
  ([#9](https://github.com/Kelet-ai/python-sdk/pull/9),
  [`6e5fcb3`](https://github.com/Kelet-ai/python-sdk/commit/6e5fcb3df92e0b813faae8093fd3dd85a02e0d0c))

- **KEL-391**: Claude Agent SDK slim observer + LoggerProvider
  ([#9](https://github.com/Kelet-ai/python-sdk/pull/9),
  [`6e5fcb3`](https://github.com/Kelet-ai/python-sdk/commit/6e5fcb3df92e0b813faae8093fd3dd85a02e0d0c))

### Refactoring

- Dedup shutdown loops (baz PR#9 feedback) ([#9](https://github.com/Kelet-ai/python-sdk/pull/9),
  [`6e5fcb3`](https://github.com/Kelet-ai/python-sdk/commit/6e5fcb3df92e0b813faae8093fd3dd85a02e0d0c))


## v1.4.2 (2026-04-21)

### Bug Fixes

- **configure**: Warn and no-op on missing credentials instead of crashing
  ([#8](https://github.com/Kelet-ai/python-sdk/pull/8),
  [`0358235`](https://github.com/Kelet-ai/python-sdk/commit/035823558e10b375ece50fad852f519a3bb7ab90))

- **configure**: Warn-and-no-op on missing KELET_API_KEY instead of crashing
  ([#8](https://github.com/Kelet-ai/python-sdk/pull/8),
  [`0358235`](https://github.com/Kelet-ai/python-sdk/commit/035823558e10b375ece50fad852f519a3bb7ab90))


## v1.4.1 (2026-04-17)

### Bug Fixes

- Drain LiteLLM callback tasks before agentic_session exit
  ([#7](https://github.com/Kelet-ai/python-sdk/pull/7),
  [`dcd672e`](https://github.com/Kelet-ai/python-sdk/commit/dcd672ef124070a08a3a9ab696dcbaba4ca87488))

- **context**: Wrap drain in try/finally so _exit always runs
  ([#7](https://github.com/Kelet-ai/python-sdk/pull/7),
  [`dcd672e`](https://github.com/Kelet-ai/python-sdk/commit/dcd672ef124070a08a3a9ab696dcbaba4ca87488))

- **drain**: Exempt pre-session tasks from aexit drain window
  ([#7](https://github.com/Kelet-ai/python-sdk/pull/7),
  [`dcd672e`](https://github.com/Kelet-ai/python-sdk/commit/dcd672ef124070a08a3a9ab696dcbaba4ca87488))

### Refactoring

- **drain**: Address code-review feedback ([#7](https://github.com/Kelet-ai/python-sdk/pull/7),
  [`dcd672e`](https://github.com/Kelet-ai/python-sdk/commit/dcd672ef124070a08a3a9ab696dcbaba4ca87488))

- **litellm**: Instrument via openinference wrapper, delete drain
  ([#7](https://github.com/Kelet-ai/python-sdk/pull/7),
  [`dcd672e`](https://github.com/Kelet-ai/python-sdk/commit/dcd672ef124070a08a3a9ab696dcbaba4ca87488))


## v1.4.0 (2026-04-16)

### Features

- Add LiteLLM and Google ADK auto-instrumentation; signal best-effort by default
  ([`10b08ce`](https://github.com/Kelet-ai/python-sdk/commit/10b08ce99ad05930c38b0f90bfd46c97f655eac2))


## v1.3.1 (2026-04-07)

### Bug Fixes

- **KEL-374**: Remove default project fallback, fail-fast on missing project
  ([#6](https://github.com/Kelet-ai/python-sdk/pull/6),
  [`92a7b09`](https://github.com/Kelet-ai/python-sdk/commit/92a7b095fe8cbfd65eee170e414e00c0d515512e))


## v1.3.0 (2026-03-23)

### Features

- **KEL-343**: Add span_processor param to configure()
  ([#5](https://github.com/Kelet-ai/python-sdk/pull/5),
  [`e6f0467`](https://github.com/Kelet-ai/python-sdk/commit/e6f04678a106b192f03c56adfe79e715b0da8ea7))


## v1.2.2 (2026-03-23)

### Bug Fixes

- **KEL-342**: Propagate metadata kwargs to child spans via SpanProcessor
  ([`5c6430d`](https://github.com/Kelet-ai/python-sdk/commit/5c6430d36e56a5fd47588cb3a66e35f7f80c0db8))


## v1.2.1 (2026-03-21)

### Bug Fixes

- Upgrade deps
  ([`2497619`](https://github.com/Kelet-ai/python-sdk/commit/249761946683f931d1d2243d1093daf0d800c918))


## v1.2.0 (2026-03-21)

### Bug Fixes

- Apply inLocalSession guard consistently and clear stale baggage keys
  ([`357f15d`](https://github.com/Kelet-ai/python-sdk/commit/357f15dac1b9a4e47e3913faa087904132e437d5))

- Apply is not None consistently and register W3C baggage propagator everywhere
  ([`5c4e758`](https://github.com/Kelet-ai/python-sdk/commit/5c4e758507f8996df83062808bded9f293650d5f))

- Register W3CBaggagePropagator in create_kelet_processor; simplify session_id guard
  ([`68ceb87`](https://github.com/Kelet-ai/python-sdk/commit/68ceb87cbce8569974f4579c08c7d539202b7b3d))

### Features

- Per-session project override and W3C baggage propagation
  ([`bba4ed2`](https://github.com/Kelet-ai/python-sdk/commit/bba4ed219c67cef52d22cabf7f46bb6c1b219065))


## v1.1.0 (2026-03-20)

### Features

- Add agent() context manager and refactor agentic_session to dual-style
  ([`a32abc4`](https://github.com/Kelet-ai/python-sdk/commit/a32abc43bb8851b40be00afae8319625a12c24ab))


## v1.0.0 (2026-03-07)

- Initial Release

## v1.0.0 (2026-03-07)

- Initial Release
