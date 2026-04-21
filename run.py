"""CLI entry point for the edge-vlm-assistant demo.

Usage:
    source ~/glasses-project/venv/bin/activate
    python run.py [--no-server]

Flags:
    --no-server   Skip launching the WebSocket metrics server (no frontend needed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.orchestrator import Orchestrator, _setup_logging
from src.server import MetricsServer


def main() -> None:
    parser = argparse.ArgumentParser(description="edge-vlm-assistant — local voice+vision assistant")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Disable the WebSocket metrics server (no frontend dashboard)",
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default="benchmarks/results/session_metrics.json",
        help="Path to save session metrics JSON on exit",
    )
    args = parser.parse_args()

    _setup_logging()
    logger = logging.getLogger(__name__)

    server: MetricsServer | None = None
    if not args.no_server:
        server = MetricsServer()
        server.start()

    orch = Orchestrator()

    # Wire server: metrics broadcast + webcam MJPEG stream
    if server is not None:
        orch.add_metrics_callback(lambda m: server.emit(m.to_dict()))
        # Frame source: latest BGR frame from the ring buffer (read-only, copy on demand)
        server.set_frame_source(lambda: orch._ring_buffer.latest())

    try:
        orch.load()
        orch.start()
        orch.run()
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
    finally:
        orch.stop()
        if server is not None:
            server.stop()

        metrics_path = Path(args.save_metrics)
        orch.save_metrics(metrics_path)

        with orch._metrics_lock:
            n = len(orch._metrics)
        if n > 0:
            print(f"\nSession complete — {n} queries. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
