import type { NextConfig } from "next";

// Static export: the UI is plain HTML/JS served from a CDN (Vercel Hobby), and
// talks to the SCRAG API over HTTPS. No server-side code, no secrets.
const config: NextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
  poweredByHeader: false,
};

export default config;
