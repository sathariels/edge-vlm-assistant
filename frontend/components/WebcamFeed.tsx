"use client";

import { useState } from "react";

const STREAM_URL = "http://localhost:8765/stream";

const s = {
  root: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 8,
  },
  label: {
    color: "var(--text-dim)",
    fontSize: 10,
    letterSpacing: "0.12em",
    textTransform: "uppercase" as const,
  },
  imgWrap: {
    position: "relative" as const,
    width: "100%",
    aspectRatio: "1 / 1",
    background: "var(--panel)",
    border: "1px solid var(--border)",
    overflow: "hidden",
  },
  img: {
    width: "100%",
    height: "100%",
    objectFit: "cover" as const,
    display: "block",
  },
  offline: {
    position: "absolute" as const,
    inset: 0,
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    color: "var(--text-mute)",
    fontSize: 11,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "var(--text-mute)",
  },
};

export default function WebcamFeed() {
  const [alive, setAlive] = useState(true);

  return (
    <div style={s.root}>
      <span style={s.label}>Webcam</span>
      <div style={s.imgWrap}>
        {alive ? (
          // MJPEG from Python backend — works as a plain img src
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={STREAM_URL}
            alt="webcam feed"
            style={s.img}
            onError={() => setAlive(false)}
          />
        ) : (
          <div style={s.offline}>
            <div style={s.dot} />
            <span>no signal</span>
            <span style={{ fontSize: 9 }}>start python run.py</span>
          </div>
        )}
      </div>
    </div>
  );
}
