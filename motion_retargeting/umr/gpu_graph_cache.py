"""Bounded retention of CUDA graph runners outside capture and replay.

Only cache references are evicted. A caller holding a runner or its tensors
keeps them alive; this module never resets a graph, empties the CUDA allocator,
or changes an in-flight solve. Byte figures are planning estimates from actual
allocator growth, not exact ownership accounting or a replacement for OOM
handling. Each backend retains at most two entries by default.
"""
from __future__ import annotations

from collections import OrderedDict
import threading

import torch


_LOCK = threading.RLock()
_BACKENDS = {}
_AGE = 0


def _input_bytes(value):
    """Conservative public tensor sizes; aliases may be counted twice."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.device.type == 'cuda' else 0
    if isinstance(value, dict):
        return sum(_input_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_input_bytes(item) for item in value)
    return 0


def _snapshot(device):
    return (int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)))


def _state(namespace, cache):
    state = _BACKENDS.get(namespace)
    if state is None:
        state = _BACKENDS[namespace] = dict(cache=cache, entries=OrderedDict(),
            hits=0, misses=0, evictions=0, clears=0, capture_failures=0,
            captured_allocated_growth_bytes=0, captured_reserved_growth_bytes=0,
            observed_input_multiplier=2., last_capture=None,
            eviction_reasons={})
    elif state['cache'] is not cache:
        raise ValueError('graph cache namespace must retain one dictionary')
    return state


def _evict(state, key, reason):
    # Removing ownership is sufficient. In particular, do not call graph.reset:
    # the caller may still hold the previous runner while obtaining a new one.
    state['cache'].pop(key, None)
    state['entries'].pop(key, None)
    state['evictions'] += 1
    reasons = state['eviction_reasons']
    reasons[reason] = reasons.get(reason, 0) + 1


def _oldest(device):
    candidates = [(item['age'], state, key)
        for state in _BACKENDS.values() for key, item in state['entries'].items()
        if item['device'] == device]
    return min(candidates, key=lambda item: item[0]) if candidates else None


def _retained_bytes(device):
    return sum(item['estimate_bytes'] for state in _BACKENDS.values()
        for item in state['entries'].values() if item['device'] == device)


def get_or_create_graph(cache, *, namespace, key, reference, factory,
                        refresh, inputs=(), max_entries=2):
    """Reuse coefficients/state or evict before constructing a missing graph.

    The process's existing Torch memory-fraction limit supplies the budget.
    Half of that limit is a soft retained-graph allowance; projected capture
    pressure also leaves ten percent headroom. The newest runner may exceed
    the soft allowance on its own. Estimates may be exceeded: real OOMs still
    propagate to the existing complete-clip split policy.

    ``factory`` captures the graph; ``refresh`` copies a hit's current inputs.
    The caller's existing per-backend lock and single-solver-thread lifecycle
    remain required. The global lock coordinates eviction across backends.
    """
    if reference.device.type != 'cuda':
        raise ValueError('CUDA graph cache requires a CUDA reference tensor')
    if max_entries not in (1, 2):
        raise ValueError('graph retention supports one or two entries per backend')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('graph cache management cannot run inside CUDA capture')
    global _AGE
    with _LOCK:
        state = _state(namespace, cache)
        runner = cache.get(key)
        if runner is not None:
            refresh(runner)
            state['hits'] += 1
            _AGE += 1
            state['entries'][key]['age'] = _AGE
            state['entries'].move_to_end(key)
            cache.move_to_end(key)
            return runner

        state['misses'] += 1
        device = str(reference.device)
        size = max(_input_bytes(inputs), reference.numel() * reference.element_size())
        estimate = max(size, int(size * state['observed_input_multiplier']))
        total = int(torch.cuda.get_device_properties(reference.device).total_memory)
        fraction = float(torch.cuda.get_per_process_memory_fraction(reference.device))
        limit = max(1, int(total * fraction))
        while len(cache) >= max_entries:
            _evict(state, next(iter(cache)), 'entry_limit_before_capture')
        # Use actual reserved/allocated pool counters for pressure. Deleting a
        # graph may leave blocks in the allocator or in an external live runner;
        # that is safe and can make this estimate conservatively evict more.
        while True:
            allocated, reserved = _snapshot(reference.device)
            over_retention = _retained_bytes(device) + estimate > limit // 2
            over_pressure = max(allocated, reserved) + estimate > int(.9 * limit)
            oldest = _oldest(device)
            if oldest is None or not (over_retention or over_pressure):
                break
            _, previous, previous_key = oldest
            _evict(previous, previous_key,
                   'byte_budget_before_capture' if over_retention else 'allocator_pressure_before_capture')

        before_allocated, before_reserved = _snapshot(reference.device)
        try:
            runner = factory()
        except Exception:
            state['capture_failures'] += 1
            raise
        after_allocated, after_reserved = _snapshot(reference.device)
        allocated_growth = max(0, after_allocated - before_allocated)
        reserved_growth = max(0, after_reserved - before_reserved)
        # The counters do not reveal exact private-pool ownership. Retain the
        # input-size estimate when reuse of existing allocator blocks masks it.
        estimate = max(estimate, allocated_growth, reserved_growth)
        state['observed_input_multiplier'] = max(state['observed_input_multiplier'],
                                                  estimate / max(1, size))
        state['captured_allocated_growth_bytes'] += allocated_growth
        state['captured_reserved_growth_bytes'] += reserved_growth
        state['last_capture'] = dict(input_tensor_bytes=size,
            allocated_growth_bytes=allocated_growth, reserved_growth_bytes=reserved_growth,
            estimated_retained_bytes=estimate, allocator_limit_bytes=limit,
            before_allocated_bytes=before_allocated, before_reserved_bytes=before_reserved,
            after_allocated_bytes=after_allocated, after_reserved_bytes=after_reserved)
        _AGE += 1
        cache[key] = runner
        state['entries'][key] = dict(device=device, estimate_bytes=estimate, age=_AGE)
        # A capture can exceed its prediction. Drop older cache references now
        # too, while always returning/retaining the newest runner itself.
        while _retained_bytes(device) > limit // 2:
            oldest = _oldest(device)
            if oldest is None:
                break
            _, previous, previous_key = oldest
            if previous is state and previous_key == key:
                break
            _evict(previous, previous_key, 'observed_growth_after_capture')
        return runner


def clear_graphs(cache, *, namespace):
    """Drop only retained references; preserve cumulative telemetry."""
    with _LOCK:
        state = _state(namespace, cache)
        state['clears'] += 1
        for key in list(cache):
            _evict(state, key, 'explicit_clear')


def graph_cache_stats():
    """Host-only cumulative counters, without allocator reads or GPU sync."""
    with _LOCK:
        return {name: dict(hits=state['hits'], misses=state['misses'],
            evictions=state['evictions'], clears=state['clears'],
            capture_failures=state['capture_failures'], retained_entries=len(state['entries']),
            estimated_retained_bytes=sum(item['estimate_bytes'] for item in state['entries'].values()),
            captured_allocated_growth_bytes=state['captured_allocated_growth_bytes'],
            captured_reserved_growth_bytes=state['captured_reserved_growth_bytes'],
            eviction_reasons=dict(state['eviction_reasons']),
            last_capture=None if state['last_capture'] is None else dict(state['last_capture']))
            for name, state in _BACKENDS.items()}
