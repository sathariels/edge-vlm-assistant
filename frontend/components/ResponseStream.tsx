"use client";

import type { QueryData } from "@/lib/websocket";

const BUDGET_MS = 800;

export default function ResponseStream({ queries }: { queries: QueryData[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%", overflow: "hidden" }}>
      <span style={{ color: "var(--text-dim)", fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", flexShrink: 0 }}>
        Response
      </span>
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
        {queries.length === 0 ? (
          <span style={{ color: "var(--text-mute)", fontSize: 11 }}>waiting…</span>
        ) : (
          queries.map((q) => {
            const over = q.stages_ms.total > BUDGET_MS;
            return (
              <div
                key={q.query_id}
                style={{
                  borderLeft: `2px solid ${over ? "var(--red)" : "var(--accent)"}`,
                  paddingLeft: 8,
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                }}
              >
                <span style={{ color: over ? "var(--red)" : "var(--accent)", fontSize: 9 }}>
                  #{q.query_id} · {q.stages_ms.total.toFixed(0)}ms {over ? "✗ OVER" : "✓"}
                </span>
                <span style={{ color: "var(--text)", fontSize: 12, lineHeight: 1.4 }}>
                  {q.response || <em style={{ color: "var(--text-mute)" }}>(empty)</em>}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
