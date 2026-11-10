import type { Metadata, Viewport } from "next";
import { Shell } from "@/components/shell";
import { Providers } from "./providers";
import "./globals.css";

const description =
  "Self-Correcting RAG: hybrid retrieval, an LLM guardrail, and a generator that is graded and corrected by an evaluator. Live demo.";

// The canonical copy of the demo. The GitHub Pages mirror points here too, so
// search engines don't index two copies. Relative metadata URLs resolve against it.
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://scrag-web.onrender.com";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: "SCRAG Demo", template: "%s · SCRAG Demo" },
  description,
  applicationName: "SCRAG",
  authors: [{ name: "Syed Waleed Ahmed", url: "https://github.com/syed-waleed-ahmed" }],
  keywords: ["RAG", "retrieval-augmented generation", "self-correction", "hybrid search", "LLM evaluation"],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: "/",
    siteName: "SCRAG",
    title: "SCRAG: Self-Correcting RAG demo",
    description,
    images: [{ url: "/og.jpg", width: 1200, height: 630, alt: "The SCRAG demo answering a sample question" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SCRAG: Self-Correcting RAG demo",
    description,
    images: ["/og.jpg"],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#020617" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans">
        <Providers>
          <Shell>{children}</Shell>
        </Providers>
      </body>
    </html>
  );
}
