import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the browser to load the MJPEG stream from the Python backend
  images: {
    remotePatterns: [{ hostname: "localhost" }],
  },
};

export default nextConfig;
