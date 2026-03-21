# CHANGELOG

<!-- version list -->

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
