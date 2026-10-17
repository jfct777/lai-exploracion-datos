#!/usr/bin/env python3
"""Dependency-free maximal-chain geometry shared by M14 and R02 H."""


def update_streaming_pair_chain(states, pair_key, position, max_gap_bp):
    """Update one pair and return a maximal chain when its gap is crossed."""
    if type(position) is not int or position <= 0 or type(max_gap_bp) is not int or max_gap_bp <= 0:
        raise ValueError("Positive integer M14 position/gap required")
    previous = states.get(pair_key)
    if previous is None:
        states[pair_key] = (position, position, 1)
        return None
    start, last, count = previous
    if position <= last:
        raise ValueError("Shared-site positions must be strictly increasing per pair")
    if position - last > max_gap_bp:
        states[pair_key] = (position, position, 1)
        return start, last, last - start + 1, count
    states[pair_key] = (start, position, count + 1)
    return None


def finish_streaming_pair_chains(states):
    """Yield and clear all remaining maximal chains in a sparse state table."""
    for pair_key, (start, last, count) in states.items():
        yield pair_key, (start, last, last - start + 1, count)
    states.clear()


class StreamingPairChains:
    """One-pair streaming form of the canonical M14 maximal-chain geometry."""

    __slots__ = ("max_gap_bp", "min_segment_bp", "min_shared_variants",
                 "start", "last", "count")

    def __init__(self, max_gap_bp, min_segment_bp, min_shared_variants):
        if not all(type(v) is int and v > 0 for v in
                   (max_gap_bp, min_segment_bp, min_shared_variants)):
            raise ValueError("Positive integer M14 chain parameters required")
        self.max_gap_bp = max_gap_bp
        self.min_segment_bp = min_segment_bp
        self.min_shared_variants = min_shared_variants
        self.start = self.last = None
        self.count = 0

    def _close(self):
        if self.count == 0:
            return None
        result = (self.start, self.last, self.last - self.start + 1, self.count)
        self.start = self.last = None
        self.count = 0
        if result[2] < self.min_segment_bp or result[3] < self.min_shared_variants:
            return None
        return result

    def push(self, position):
        state = {} if self.last is None else {0: (self.start, self.last, self.count)}
        closed = update_streaming_pair_chain(state, 0, position, self.max_gap_bp)
        self.start, self.last, self.count = state[0]
        if closed is None:
            return None
        if closed[2] < self.min_segment_bp or closed[3] < self.min_shared_variants:
            return None
        return closed

    def finish(self):
        return self._close()
