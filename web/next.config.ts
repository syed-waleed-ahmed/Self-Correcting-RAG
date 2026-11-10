import type { NextConfig } from "next";

// Static export: the UI is plain HTML/JS served from a CDN (GitHub Pages or
// Vercel), and talks to the SCRAG API over HTTPS. No server-side code, no secrets.
//
// NEXT_PUBLIC_BASE_PATH is set when the site is served from a sub-path, e.g.
// "/Self-Correcting-RAG" on GitHub Pages; leave it unset for Vercel or locally.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;

const config: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  reactStrictMode: true,
  images: { unoptimized: true },
  poweredByHeader: false,
};

export default config;
