"""Record real pipeline runs for the demo's sample questions.

The public demo serves these recorded results for its sample questions, always
labelled ``cached`` with the recording timestamp, so first-time visitors don't
spend the free-tier LLM quota. They are genuine executions, not written by hand.
Re-run this whenever the sample corpus (``data/docs``) or the models change:
answers are only served while the corpus fingerprint matches.

Usage (needs GROQ_API_KEY; spends roughly 4-8K tokens per question):

    python scripts/record_samples.py                  # writes data/demo/samples.json
    python scripts/record_samples.py --pause 40       # slower, for tight TPM limits
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrag.api.routes import _to_response  # noqa: E402
from scrag.config import Settings  # noqa: E402
from scrag.container import build_container  # noqa: E402

QUESTIONS = [
    "What triggers self-correction, and what does the pipeline return?",
    "How does hybrid search combine keyword and vector results?",
    "Why is a high evaluator score not proof that an answer is correct?",
    "How can prompt injection reach a RAG system, and how is it mitigated?",
    "Why does the demo use the ONNX embedding backend on small instances?",
    "What does the BM25 parameter k1 control?",
    "What does an API error response contain?",
    "Why is the sample knowledge base built into the container image?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "demo" / "samples.json")
    parser.add_argument("--pause", type=float, default=30.0, help="seconds between questions")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as index_dir:
        settings = Settings(
            docs_dir=ROOT / "data" / "docs",
            index_dir=Path(index_dir),
            samples_path=Path(index_dir) / "none.json",  # never serve old samples while recording
            answer_cache_size=0,
        )
        top_k = settings.demo_top_k  # what the live demo uses
        container = build_container(settings)
        if container.pipeline is None:
            print("No LLM key configured (GROQ_API_KEY).", file=sys.stderr)
            return 1
        container.run_ingest(reset=True)

        items = []
        for i, question in enumerate(QUESTIONS):
            if i:
                time.sleep(args.pause)
            start = time.perf_counter()
            result = container.pipeline.run(question, top_k=top_k)
            response = _to_response(result, settings.llm_prices).model_dump(
                exclude={"cached", "cached_at", "cache_source", "conversation_id", "history_saved"}
            )
            print(
                f"[{i + 1}/{len(QUESTIONS)}] {result.evaluation_status} "
                f"score={result.evaluation_score} attempts={result.attempts} "
                f"({time.perf_counter() - start:.1f}s) {question}"
            )
            items.append({"question": question, "response": response})

        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": settings.generator_model,
            "retrieval_mode": settings.retrieval_mode,
            "top_k": top_k,
            "corpus_fingerprint": container.corpus_fingerprint(),
            "items": items,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} recorded answers to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
