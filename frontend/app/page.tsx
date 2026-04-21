"use client";

import dynamic from "next/dynamic";
import { useMetricsSocket } from "@/lib/websocket";

// Dynamic imports prevent SSR for components that use browser APIs
const WebcamFeed      = dynamic(() => import("@/components/WebcamFeed"),      { ssr: false });
const LatencyWaterfall = dynamic(() => import("@/components/LatencyWaterfall"), { ssr: false });
const TranscriptStream = dynamic(() => import("@/components/TranscriptStream"), { ssr: false });
const ResponseStream   = dynamic(() => import("@/components/ResponseStream"),   { ssr: false });

const STATUS_COLOR: Record<string, string> = {
  connected:    "var(--accent)",
  disconnected: "var(--red)",
  connecting:   "var(--col-vad)",
};

export default function Dashboard() {
  const { queries, status } = useMetricsSocket();
  const latestTotal = queries[0]?.stages_ms.total;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateRows: "auto 1fr auto",
        height: "100vh",
        gap: 0,
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 24,
          padding: "8px 16px",
          borderBottom: "1px solid var(--border)",
          background: "var(--panel)",
          flexWrap: "wrap" as const,
        }}
      >
        <span style={{ color: "var(--accent)", fontWeight: 700, letterSpacing: "0.08em", fontSize: 13 }}>
          EDGE-VLM
        </span>
        <span style={{ color: "var(--text-dim)", fontSize: 10 }}>GLASSES MODE</span>

        {/* connection status */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: STATUS_COLOR[status] ?? "var(--text-dim)",
            }}
          />
          <span style={{ color: STATUS_COLOR[status] ?? "var(--text-dim)", fontSize: 10, textTransform: "uppercase" }}>
            {status}
          </span>
        </div>

        <span style={{ color: "var(--text-dim)", fontSize: 10 }}>
          {queries.length} queries
        </span>

        {latestTotal !== undefined && (
          <span
            style={{
              marginLeft: "auto",
              color: latestTotal > 800 ? "var(--red)" : "var(--accent)",
              fontSize: 13,
              fontWeight: 700,
            }}
          >
            {latestTotal.toFixed(0)}ms {latestTotal > 800 ? "✗" : "✓"}
          </span>
        )}
      </header>

      {/* ── Main ───────────────────────────────────────────────────── */}
      <main
        style={{
          display: "grid",
          gridTemplateColumns: "220px 1fr",
          gridTemplateRows: "1fr 1fr",
          gap: 1,
          background: "var(--border)",
          overflow: "hidden",
          minHeight: 0,
        }}
      >
        {/* Webcam — spans both rows on the left */}
        <div
          style={{
            gridRow: "1 / 3",
            background: "var(--panel)",
            padding: 12,
            overflow: "hidden",
          }}
        >
          <WebcamFeed />
        </div>

        {/* Waterfall — top right */}
        <div
          style={{
            background: "var(--panel)",
            padding: 12,
            overflow: "hidden",
            minHeight: 0,
          }}
        >
          <LatencyWaterfall queries={queries} />
        </div>

        {/* Transcript + Response — bottom right, split 50/50 */}
        <div
          style={{
            background: "var(--panel)",
            padding: 12,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            overflow: "hidden",
            minHeight: 0,
          }}
        >
          <TranscriptStream queries={queries} />
          <ResponseStream   queries={queries} />
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <footer
        style={{
          display: "flex",
          alignItems: "center",
          gap: 20,
          padding: "4px 16px",
          borderTop: "1px solid var(--border)",
          background: "var(--panel)",
          color: "var(--text-mute)",
          fontSize: 9,
          letterSpacing: "0.1em",
        }}
      >
        <span>MOONDREAM 2 MLX · DISTIL-WHISPER · PIPER TTS · SILERO VAD</span>
        <span style={{ marginLeft: "auto" }}>ws://localhost:8765</span>
      </footer>
    </div>
  );
}
