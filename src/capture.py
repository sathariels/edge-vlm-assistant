"""Loop 1 — Webcam capture thread.

Continuously reads from the webcam, downsamples frames to VLM-native
resolution, and pushes them into the ring buffer.

Design constraints:
- Dedicated thread; never blocks the inference loop.
- Fixed-resolution output — never passes raw 1080p to VLM.
- Pre-allocates the resize target to avoid per-frame heap allocation.

Phase 1 standalone usage:
    python -m src.capture
  Opens webcam, displays live feed, prints per-100-frame latency stats.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.config import (
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    RING_BUFFER_SIZE,
)
from src.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class CaptureLoop:
    """Webcam → downsample → ring buffer, running in a daemon thread.

    Phase 3 optimization — pre-allocated frame pool:
      RING_BUFFER_SIZE numpy arrays are allocated once at construction.
      cv2.resize writes directly into pool[pool_idx] via the dst= parameter,
      eliminating the per-frame malloc that Phase 2 incurred.
      pool_idx advances in lockstep with the ring buffer head, so by the time
      a pool slot is reused the ring buffer has already evicted that slot.
      The inference thread copies the frame on grab (one alloc per query, not
      per frame) so there is no race with the overwrite.
    """

    def __init__(
        self,
        ring_buffer: Optional[RingBuffer[np.ndarray]] = None,
        camera_index: int = 0,
    ) -> None:
        self._buffer: RingBuffer[np.ndarray] = ring_buffer or RingBuffer(RING_BUFFER_SIZE)
        self._camera_index = camera_index
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count: int = 0
        self._capture_latencies: list[float] = []

        # Pre-allocated frame pool — no per-frame heap allocation at 30fps
        self._pool: list[np.ndarray] = [
            np.empty((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
            for _ in range(RING_BUFFER_SIZE)
        ]
        self._pool_idx: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="capture-loop"
        )
        self._thread.start()
        logger.info("CaptureLoop started (camera=%d)", self._camera_index)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("CaptureLoop stopped after %d frames", self._frame_count)

    # ------------------------------------------------------------------
    # Ring buffer access
    # ------------------------------------------------------------------

    @property
    def ring_buffer(self) -> RingBuffer[np.ndarray]:
        return self._buffer

    def latest_frame(self) -> Optional[np.ndarray]:
        return self._buffer.latest()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera %d", self._camera_index)
            return

        # Request native capture res; actual res depends on hardware
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

        logger.info(
            "Camera opened: %dx%d @ %.0ffps (requested %d)",
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            CAPTURE_FPS,
        )

        try:
            while not self._stop_event.is_set():
                t_read_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.01)
                    continue

                # Downsample to VLM-native resolution (448×448 by default).
                # INTER_AREA is best quality for shrinking.
                # Write directly into pre-allocated pool slot — no malloc.
                dst = self._pool[self._pool_idx]
                cv2.resize(
                    frame,
                    (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                    dst=dst,
                    interpolation=cv2.INTER_AREA,
                )
                t_ready = time.perf_counter()

                self._buffer.push(dst)
                self._pool_idx = (self._pool_idx + 1) % RING_BUFFER_SIZE
                self._frame_count += 1

                latency_ms = (t_ready - t_read_start) * 1000.0
                self._capture_latencies.append(latency_ms)

                if self._frame_count % 100 == 0:
                    recent = self._capture_latencies[-100:]
                    logger.info(
                        "Capture: %d frames | latency p50=%.1fms p95=%.1fms",
                        self._frame_count,
                        statistics.median(recent),
                        sorted(recent)[int(len(recent) * 0.95)],
                    )
        finally:
            cap.release()
            logger.info("Camera released")


# ---------------------------------------------------------------------------
# Phase 1 standalone demo
# ---------------------------------------------------------------------------

def _run_phase1_demo() -> None:
    """Open webcam, show live feed, print latency stats. Press 'q' to quit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    buf: RingBuffer[np.ndarray] = RingBuffer(RING_BUFFER_SIZE)
    loop = CaptureLoop(ring_buffer=buf)
    loop.start()

    print(f"Webcam feed — {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}. Press 'q' to quit.")
    frame_times: list[float] = []
    t_last = time.perf_counter()

    try:
        while True:
            frame = buf.latest()
            if frame is None:
                time.sleep(0.005)
                continue

            now = time.perf_counter()
            frame_times.append(now - t_last)
            t_last = now

            # Overlay FPS and latency
            if len(frame_times) > 10:
                fps = 1.0 / statistics.mean(frame_times[-30:])
                display = frame.copy()
                cv2.putText(
                    display,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("edge-vlm-assistant [capture]", display)
            else:
                cv2.imshow("edge-vlm-assistant [capture]", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        loop.stop()
        cv2.destroyAllWindows()

    if frame_times:
        fps_vals = [1.0 / t for t in frame_times if t > 0]
        print(f"\nCapture stats over {len(fps_vals)} frames:")
        print(f"  avg FPS : {statistics.mean(fps_vals):.1f}")
        print(f"  p50 FPS : {statistics.median(fps_vals):.1f}")
        print(f"  min FPS : {min(fps_vals):.1f}")


if __name__ == "__main__":
    _run_phase1_demo()
