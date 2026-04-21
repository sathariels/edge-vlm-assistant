"use client";

import type { QueryData } from "@/lib/websocket";

export default function TranscriptStream({ queries }: { queries: QueryData[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%", overflow: "hidden" }}>
      <span style={{ color: "var(--text-dim)", fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", flexShrink: 0 }}>
        Transcript
      </span>
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
        {queries.length === 0 ? (
          <span style={{ color: "var(--text-mute)", fontSize: 11 }}>waiting…</span>
        ) : (
          queries.map((q) => (
            <div
              key={q.query_id}
              style={{
                borderLeft: "2px solid var(--col-asr)",
                paddingLeft: 8,
                display: "flex",
                flexDirection: "column",
                gap: 2,
              }}
            >
              <span style={{ color: "var(--text-dim)", fontSize: 9 }}>#{q.query_id} · {q.stages_ms.asr.toFixed(0)}ms ASR</span>
              <span style={{ color: "var(--text)", fontSize: 12, lineHeight: 1.4 }}>
                &ldquo;{q.transcript}&rdquo;
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
