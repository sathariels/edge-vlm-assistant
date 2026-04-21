"use client";

import type { QueryData } from "@/lib/websocket";

// ── Constants ────────────────────────────────────────────────────────────────
const BUDGET_MS = 800;
const SCALE_MS  = 1100; // px-width of the timeline (1100ms shown)
const ROW_H     = 18;   // px height of each bar row
const ROW_GAP   = 3;    // px gap between rows
const MAX_ROWS  = 15;

const STAGE_COLOR = {
  asr: "var(--col-asr)",
  vlm: "var(--col-vlm)",
  tts: "var(--col-tts)",
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function pct(ms: number): string {
  return `${(ms / SCALE_MS) * 100}%`;
}

// ── Sub-components ───────────────────────────────────────────────────────────
function Bar({
  startMs,
  durationMs,
  color,
  label,
}: {
  startMs: number;
  durationMs: number;
  color: string;
  label: string;
}) {
  if (durationMs <= 0) return null;
  const clampedDur = Math.min(durationMs, SCALE_MS - startMs);
  return (
    <div
      title={`${label}: ${durationMs.toFixed(0)}ms`}
      style={{
        position: "absolute",
        left: pct(startMs),
        width: pct(clampedDur),
        height: "100%",
        background: color,
        opacity: 0.82,
        borderRadius: 2,
      }}
    />
  );
}

function WaterfallRow({ q }: { q: QueryData }) {
  const { stages_ms: s, waterfall: w } = q;
  const over = s.total > BUDGET_MS;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        height: ROW_H,
        marginBottom: ROW_GAP,
      }}
    >
      {/* Query ID */}
      <span
        style={{
          width: 28,
          textAlign: "right",
          color: "var(--text-dim)",
          fontSize: 10,
          flexShrink: 0,
        }}
      >
        #{q.query_id}
      </span>

      {/* Bar timeline */}
      <div style={{ position: "relative", flex: 1, height: "100%" }}>
        <Bar startMs={w.asr_start_ms}  durationMs={s.asr}             color={STAGE_COLOR.asr} label="ASR" />
        <Bar startMs={w.vlm_start_ms}  durationMs={s.vlm_first_token} color={STAGE_COLOR.vlm} label="VLM" />
        <Bar startMs={w.tts_start_ms}  durationMs={s.tts_first_chunk} color={STAGE_COLOR.tts} label="TTS" />

        {/* Budget marker */}
        <div
          style={{
            position: "absolute",
            left: pct(BUDGET_MS),
            top: 0,
            width: 1,
            height: "100%",
            background: "var(--red)",
            opacity: 0.5,
          }}
        />
      </div>

      {/* Total */}
      <span
        style={{
          width: 52,
          textAlign: "right",
          color: over ? "var(--red)" : "var(--accent)",
          fontSize: 11,
          flexShrink: 0,
        }}
      >
        {s.total.toFixed(0)}ms
      </span>
    </div>
  );
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  const items = [
    { label: "ASR",     color: STAGE_COLOR.asr },
    { label: "VLM",     color: STAGE_COLOR.vlm },
    { label: "TTS 1st", color: STAGE_COLOR.tts },
    { label: `${BUDGET_MS}ms budget`, color: "var(--red)", dash: true },
  ];
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" as const }}>
      {items.map(({ label, color, dash }) => (
        <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <div
            style={{
              width: dash ? 12 : 10,
              height: dash ? 1 : 10,
              background: color,
              opacity: 0.82,
              borderStyle: dash ? "dashed" : "solid",
              borderRadius: dash ? 0 : 2,
              flexShrink: 0,
            }}
          />
          <span style={{ color: "var(--text-dim)", fontSize: 10 }}>{label}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function LatencyWaterfall({ queries }: { queries: QueryData[] }) {
  const rows = queries.slice(0, MAX_ROWS);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%" }}>
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ color: "var(--text-dim)", fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase" }}>
          Latency Waterfall
        </span>
        <Legend />
      </div>

      {/* Timeline axis labels */}
      <div style={{ display: "flex", paddingLeft: 36 }}>
        <div style={{ position: "relative", flex: 1 }}>
          {[0, 200, 400, 600, 800, 1000].map((ms) => (
            <span
              key={ms}
              style={{
                position: "absolute",
                left: pct(ms),
                color: ms === BUDGET_MS ? "var(--red)" : "var(--text-mute)",
                fontSize: 9,
                transform: "translateX(-50%)",
              }}
            >
              {ms}
            </span>
          ))}
        </div>
        <div style={{ width: 60 }} />
      </div>

      {/* Rows */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {rows.length === 0 ? (
          <div style={{ color: "var(--text-mute)", fontSize: 11, paddingTop: 20, textAlign: "center" }}>
            waiting for queries…
          </div>
        ) : (
          rows.map((q) => <WaterfallRow key={q.query_id} q={q} />)
        )}
      </div>
    </div>
  );
}
