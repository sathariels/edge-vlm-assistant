import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "edge-vlm-assistant",
  description: "On-device voice+vision assistant — latency debug console",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
