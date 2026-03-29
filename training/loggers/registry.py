"""
MetricsRegistry — decoupled, main-thread-safe custom metric collection.

Components register a callable that returns a flat dict of metric names
to scalar values. The main training loop calls collect() at each logging
point, which invokes every provider and returns a single merged dict
with namespaced keys.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any, Callable, Dict


class MetricsRegistry:
    """
    Collects custom metrics from registered providers for main-thread logging.

    Usage
    -----
        registry = MetricsRegistry()
        registry.register('opponent_pool', pool.get_metrics)
        registry.register('reward', reward_fn.get_metrics)

        # In the main-thread logging block:
        metrics = registry.collect()
        logger.log(step, **metrics)

    Provider contract
    -----------------
    Each provider callable must:
      - Accept no arguments: ``def get_metrics() -> dict[str, float | int]``
      - Return a flat dict of metric names to scalar values
      - Be safe to call from the main thread (read-only access to shared state)
    """

    def __init__(self) -> None:
        self._providers: OrderedDict[str, Callable[[], Dict[str, Any]]] = OrderedDict()

    def register(self, namespace: str, provider: Callable[[], Dict[str, Any]]) -> None:
        """Register a metrics provider under the given namespace."""
        self._providers[namespace] = provider

    def unregister(self, namespace: str) -> None:
        """Remove a previously registered provider."""
        self._providers.pop(namespace, None)

    def collect(self) -> Dict[str, Any]:
        """Call all providers and return a merged, namespaced metrics dict."""
        merged: Dict[str, Any] = {}
        for namespace, provider in self._providers.items():
            try:
                raw = provider()
            except Exception as exc:
                print(f'[metrics] Provider {namespace!r} failed: {exc}',
                      file=sys.stderr)
                continue
            for key, value in raw.items():
                merged[f'{namespace}/{key}'] = value
        return merged

    @property
    def namespaces(self) -> list[str]:
        """Return list of registered namespace names."""
        return list(self._providers.keys())
