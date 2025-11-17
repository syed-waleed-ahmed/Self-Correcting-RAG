import argparse

from src.retriever import build_index
from src.pipeline import SelfCorrectingRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Self-Correcting RAG Pipeline")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the vector index from documents and exit.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask the RAG system.",
    )

    args = parser.parse_args()

    if args.build_index:
        build_index()
        return

    if not args.query:
        print("Please provide --query 'your question' or --build-index.")
        return

    pipeline = SelfCorrectingRAGPipeline()
    result = pipeline.run(args.query)

    print("\n=== Final Answer ===")
    print(result.answer)
    print("\n=== Evaluation ===")
    print(f"Score: {result.evaluation_score:.2f}")
    print(f"Explanation: {result.evaluation_explanation}")
    print(f"Attempts: {result.attempts}")

    print("\n=== Context Chunks Used ===")
    for i, doc in enumerate(result.used_docs, start=1):
        print(
            f"\n[Chunk {i} from {doc['filename']}, "
            f"guardrail={doc.get('guardrail_score', 0):.2f}]"
        )
        text = doc["text"]
        print(text[:400] + ("..." if len(text) > 400 else ""))


if __name__ == "__main__":
    main()
