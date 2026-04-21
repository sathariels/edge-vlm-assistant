"""Fixed-size thread-safe ring buffer for webcam frames.

Design notes:
- Fixed allocation at construction — no per-push heap allocation.
- Single writer (capture thread), potentially multiple readers (inference thread).
- Uses a Lock for correctness; the critical section is O(1) so contention is negligible.
- `latest()` returns the most recently pushed item without consuming it.
"""

from __future__ import annotations

import threading
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-size circular buffer. Thread-safe for concurrent push/latest."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"RingBuffer size must be >= 1, got {size}")
        self._size = size
        self._buffer: list[Optional[T]] = [None] * size
        self._head: int = 0   # index of the *next* write slot
        self._count: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def push(self, item: T) -> None:
        """Push an item into the buffer, overwriting the oldest entry when full."""
        with self._lock:
            self._buffer[self._head] = item
            self._head = (self._head + 1) % self._size
            if self._count < self._size:
                self._count += 1

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def latest(self) -> Optional[T]:
        """Return the most recently pushed item without removing it.

        Returns None if the buffer is empty.
        """
        with self._lock:
            if self._count == 0:
                return None
            idx = (self._head - 1) % self._size
            return self._buffer[idx]

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def __repr__(self) -> str:
        return f"RingBuffer(size={self._size}, count={len(self)})"
