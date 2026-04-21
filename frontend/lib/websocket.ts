"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface WaterfallOffsets {
  asr_start_ms: number;
  vlm_start_ms: number;
  tts_start_ms: number;
}

export interface StagesMs {
  vad: number;
  asr: number;
  vlm_first_token: number;
  tts_first_chunk: number;
  total: number;
}

export interface QueryData {
  query_id: number;
  transcript: string;
  response: string;
  stages_ms: StagesMs;
  waterfall: WaterfallOffsets;
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

const WS_URL = "ws://localhost:8765/ws";
const MAX_HISTORY = 30;
const RECONNECT_DELAY_MS = 2000;

export function useMetricsSocket() {
  const [queries, setQueries] = useState<QueryData[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    // Guard against double-connect
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setStatus("connected");

      ws.onclose = () => {
        setStatus("disconnected");
        timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      };

      ws.onerror = () => ws.close();

      ws.onmessage = (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data as string) as QueryData;
          setQueries((prev) => [data, ...prev].slice(0, MAX_HISTORY));
        } catch {
          // Ignore malformed messages
        }
      };
    } catch {
      setStatus("disconnected");
      timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [connect]);

  return { queries, status };
}
