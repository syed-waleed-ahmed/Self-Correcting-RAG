"""Self-Correcting RAG: a production-grade multi-agent RAG pipeline.

Public API:
    from scrag import Settings, get_settings
    from scrag import SelfCorrectingRAGPipeline, PipelineResult
    from scrag import build_container
"""

from __future__ import annotations

from .config import Settings, get_settings
from .container import Container, build_container
from .models import Chunk, PipelineResult, RetrievedChunk
from .pipeline import SelfCorrectingRAGPipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "get_settings",
    "Container",
    "build_container",
    "Chunk",
    "RetrievedChunk",
    "PipelineResult",
    "SelfCorrectingRAGPipeline",
]
