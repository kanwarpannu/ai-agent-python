"""Evaluation harness for the RAG pipeline.

Scores ``rag_essentials`` runs through three lenses:

- **black-box** — judge the final answer given only the question (+ reference).
- **single-step** — score the retrieval step and the generation step in isolation.
- **trajectory** — verify the sequence of internal state transitions.
"""

from .runner import run_eval

__all__ = ["run_eval"]
