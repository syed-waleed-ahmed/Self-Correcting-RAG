import type { Metadata } from "next";
import { Card } from "@/components/ui";

export const metadata: Metadata = {
  title: "How it works",
  description: "SCRAG's pipeline, measured retrieval quality, free-tier deployment and privacy.",
};

const REPO = "https://github.com/syed-waleed-ahmed/Self-Correcting-RAG";

const STAGES = [
  {
    name: "Hybrid retrieval",
    text: "BM25 keyword ranking and dense embeddings (all-MiniLM-L6-v2 via ONNX) are fused with Reciprocal Rank Fusion, so exact identifiers and paraphrases are both found.",
  },
  {
    name: "Guardrail",
    text: "An LLM scores every retrieved chunk for relevance and drops off-topic context before it can mislead the answer. If it rejects everything, raw retrieval is used.",
  },
  {
    name: "Generate",
    text: "The generator answers only from the kept context.",
  },
  {
    name: "Evaluate and self-correct",
    text: "An evaluator grades how well the answer is supported. Below the threshold, its critique goes back to the generator for another attempt; the best-scoring attempt is returned.",
  },
];

// Measured 2026-11-10 on the 40-question golden set (docs/evaluation.md).
const RESULTS = [
  { k: 5, mode: "dense", recall: "0.950", hit: "0.975", mrr: "0.915", ndcg: "0.908" },
  { k: 5, mode: "hybrid", recall: "0.988", hit: "1.000", mrr: "0.961", ndcg: "0.957" },
];

export default function AboutPage() {
  return (
    <div className="space-y-6">
      <Card title="The pipeline">
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {STAGES.map((stage, i) => (
            <li key={stage.name} className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
              <p className="text-xs font-medium text-slate-400">Step {i + 1}</p>
              <p className="font-medium">{stage.name}</p>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{stage.text}</p>
            </li>
          ))}
        </ol>
        <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">
          The <span className="font-medium">Agent</span> tab adds a bounded planner: it picks tools (document search, a
          safe calculator), runs them in parallel, writes an answer that cites its evidence, and verifies it, within
          hard limits on steps, LLM calls and time.
        </p>
      </Card>

      <Card title="Measured retrieval quality">
        <p className="mb-3 text-sm text-slate-600 dark:text-slate-300">
          40 labelled questions over the 16 sample documents, at k = 5. Every number was measured (2026-11-10) and can
          be reproduced with <code className="font-mono text-xs">scrag eval</code>.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="py-1.5 pr-4 font-medium">Retrieval</th>
                <th className="py-1.5 pr-4 font-medium">Recall@5</th>
                <th className="py-1.5 pr-4 font-medium">Hit@5</th>
                <th className="py-1.5 pr-4 font-medium">MRR</th>
                <th className="py-1.5 font-medium">nDCG@5</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {RESULTS.map((row) => (
                <tr key={row.mode} className={row.mode === "hybrid" ? "font-semibold text-brand-700 dark:text-brand-100" : ""}>
                  <td className="py-1.5 pr-4 font-sans">{row.mode === "hybrid" ? "Hybrid (default)" : "Dense only"}</td>
                  <td className="py-1.5 pr-4">{row.recall}</td>
                  <td className="py-1.5 pr-4">{row.hit}</td>
                  <td className="py-1.5 pr-4">{row.mrr}</td>
                  <td className="py-1.5">{row.ndcg}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          Hybrid search helps most on questions with exact terms (lexical MRR 0.838 → 0.914) at the same latency (about
          4 ms). Two questions remain known misses and stay in the set.{" "}
          <a href={`${REPO}/blob/main/docs/evaluation.md`} className="text-brand-600 hover:underline dark:text-brand-100">
            Methodology and full results
          </a>
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Runs on free tiers">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-600 dark:text-slate-300">
            <li>API: FastAPI in Docker on Render Free (512 MB, sleeps when idle; the first request wakes it).</li>
            <li>Database: Neon Free Postgres with pgvector, for sessions, uploads, history and jobs.</li>
            <li>Web UI: a static export on a CDN. No secrets in the browser.</li>
            <li>LLM: Groq&apos;s free tier, protected by per-visitor rate limits and a daily budget.</li>
          </ul>
        </Card>
        <Card title="Privacy and limits">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-600 dark:text-slate-300">
            <li>No accounts and no tracking cookies. A session is an anonymous token kept in your browser.</li>
            <li>Sessions, uploads and history are deleted after 24 hours, or at once with “Delete all my data”.</li>
            <li>Only extracted text is stored, and other sessions can never read it.</li>
            <li>Questions and document text are sent to the LLM provider: don&apos;t use personal data.</li>
          </ul>
        </Card>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-300">
        Source code, architecture decisions and the roadmap are on{" "}
        <a href={REPO} className="text-brand-600 hover:underline dark:text-brand-100">
          GitHub
        </a>{" "}
        (MIT licensed).
      </p>
    </div>
  );
}
