from __future__ import annotations


def vector_literal(vec: list[float]) -> str:
    # pgvector input format: '[1,2,3]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
