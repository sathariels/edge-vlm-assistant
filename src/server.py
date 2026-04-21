"""FastAPI + WebSocket bridge to the Next.js frontend.

Pushes per-query QueryMetrics waterfall JSON to all connected clients.
Runs in a background daemon thread so it doesn't block the inference loops.

Start:
    server = MetricsServer()
    server.start()          # launches uvicorn in a background thread

Emit from any thread:
    server.emit(metrics.to_dict())

Frontend connects to:
    ws://localhost:8765/ws

Phase 2 implementation:
  - WebSocket /ws  — waterfall JSON broadcast per query
  - GET /health    — liveness check

Phase 4 additions (not yet implemented):
  - JPEG frame stream for frontend webcam display
  - Streaming VLM token push
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


class MetricsServer:
    """WebSocket server for the frontend dashboard.

    Thread-safe: emit() can be called from any thread (e.g. the inference thread).
    The FastAPI app and asyncio loop run in a dedicated daemon thread.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue[dict[str, Any]]] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: set[Any] = set()
        # Optional frame source for MJPEG stream — set via set_frame_source()
        self._frame_source: Optional[Callable[[], Optional[np.ndarray]]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch uvicorn in a background daemon thread."""
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_server, args=(ready,), daemon=True, name="metrics-server"
        )
        self._thread.start()
        ready.wait(timeout=5.0)
        logger.info("MetricsServer listening on ws://%s:%d/ws", self._host, self._port)

    def stop(self) -> None:
        # uvicorn in a daemon thread will stop when the main process exits.
        # For a clean stop, signal the loop.
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # Emit from any thread
    # ------------------------------------------------------------------

    def set_frame_source(self, fn: Callable[[], Optional[np.ndarray]]) -> None:
        """Register a callable that returns the latest BGR webcam frame (or None).

        Called from the main thread before start(). The MJPEG /stream endpoint
        will invoke this from the asyncio event loop via run_in_executor so it
        never blocks the loop.
        """
        self._frame_source = fn

    def emit(self, data: dict[str, Any]) -> None:
        """Broadcast *data* to all connected WebSocket clients.

        Safe to call from any thread (inference thread, benchmark, etc.).
        Silently no-ops if no clients are connected or server not yet started.
        """
        if self._loop is None or self._queue is None:
            return
        if not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._queue.put(data), self._loop)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_server(self, ready: threading.Event) -> None:
        """Entry point for the server daemon thread.

        Uses uvloop when available (Phase 3 optimization) for lower-overhead
        asyncio. Falls back to the default asyncio event loop silently.
        """
        import uvicorn  # type: ignore[import-untyped]

        try:
            import uvloop  # type: ignore[import-untyped]
            loop: asyncio.AbstractEventLoop = uvloop.new_event_loop()
            logger.debug("Server using uvloop")
        except ImportError:
            loop = asyncio.new_event_loop()
            logger.debug("Server using default asyncio loop (install uvloop for lower latency)")

        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue()

        app = self._build_app()

        config = uvicorn.Config(
            app=app,
            host=self._host,
            port=self._port,
            loop="none",       # we manage our own loop
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        async def _serve() -> None:
            asyncio.create_task(self._broadcast_worker())
            ready.set()
            await server.serve()

        loop.run_until_complete(_serve())

    def _build_app(self) -> Any:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore[import-untyped]
        from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-untyped]
        from fastapi.responses import StreamingResponse  # type: ignore[import-untyped]

        app = FastAPI(title="edge-vlm-assistant metrics")

        # Allow the Next.js dev server (localhost:3000) to connect
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/stream")
        async def mjpeg_stream() -> Any:
            """MJPEG stream of the latest webcam frame at ~10fps.

            Browser usage:  <img src="http://localhost:8765/stream" />
            """
            import cv2  # type: ignore[import-untyped]

            async def _generate() -> Any:
                loop = asyncio.get_event_loop()
                while True:
                    if self._frame_source is not None:
                        frame: Optional[np.ndarray] = await loop.run_in_executor(
                            None, self._frame_source
                        )
                        if frame is not None:
                            _, buf = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                            )
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + buf.tobytes()
                                + b"\r\n"
                            )
                    await asyncio.sleep(0.1)  # ~10fps — enough for a debug dashboard

            return StreamingResponse(
                _generate(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            self._clients.add(ws)
            logger.debug("WebSocket client connected (%d total)", len(self._clients))
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(ws)
                logger.debug("WebSocket client disconnected (%d remaining)", len(self._clients))

        return app

    async def _broadcast_worker(self) -> None:
        """Drain the emit queue and fan-out to all WebSocket clients."""
        assert self._queue is not None
        while True:
            data = await self._queue.get()
            dead: set[Any] = set()
            for ws in list(self._clients):
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.add(ws)
            self._clients -= dead
