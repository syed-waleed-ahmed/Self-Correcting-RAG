# 0001: fastembed (ONNX) is the deployment embedder

**Status:** Accepted (2026-11-10).

**Context.** The free API host (Render Free) has 512 MB of RAM and 0.1 CPU.
The original embedder loads PyTorch through sentence-transformers.

**Evidence (2026-11-10).** Same model (`all-MiniLM-L6-v2`), measured in fresh
processes:

| Backend | Peak memory | Per-query time |
| --- | --- | --- |
| fastembed (ONNX) | 211 MB | 3.0 ms |
| sentence-transformers (torch) | 428 MB | 12.3 ms |

On every test text, the cosine similarity between the two backends' vectors
was 1.00000. See `docs/evaluation.md`.

**Decision.**
- Add `FastEmbedEmbedder` and `SCRAG_EMBEDDING_BACKEND` (`auto` by default:
  fastembed when installed, otherwise sentence-transformers).
- The Docker image installs only the `onnx` extra, with no torch.

**Verified (2026-11-10, CI run 42735463057).** The deployment image ran under
`--memory=512m --cpus=0.1` with a peak of 210 MB and was not OOM-killed. The
image is 296 MB.

**Consequences.**
- Existing indexes stay valid, because the vectors are identical.
- torch remains available through the `local` extra.
- The CI `container` job checks peak memory under a 512 MB limit on every PR.
