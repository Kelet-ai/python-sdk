"""Detect whether code is executing inside a Temporal workflow sandbox.

Used by _context.py and _signal.py to branch between full mode (OTEL attach,
background drain, direct httpx) and workflow mode (contextvars only, dispatch
via Temporal activity).

Lives in its own module so both _context.py and _signal.py can import it
without creating a cycle. Has no imports from `kelet` itself.
"""


def in_temporal_workflow() -> bool:
    """Return True iff the current call stack is inside a Temporal workflow.

    Falls back to False if `temporalio` is not installed or the runtime check
    raises for any reason. Never raises.
    """
    try:
        from temporalio import workflow  # type: ignore[reportMissingImports]

        return workflow.in_workflow()
    except Exception:
        return False
