"""Kelet SDK framework integrations.

This package holds optional, per-framework instrumentors that are only
imported when the corresponding third-party library is installed. Each
sub-package exposes a ``BaseInstrumentor`` subclass that can be invoked
from ``kelet._configure._auto_instrument_frameworks``.
"""
