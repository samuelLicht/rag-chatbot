import json
from pathlib import Path

from rag.embeddings import create_embeddings
from rag.vector_store import search_index


def load_chunks(path: Path) -> list[dict]:
    """Load processed chunks from a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def retrieve_context(
    query: str,
    chunks: list[dict],
    index,
    embedding_model,
    top_k: int = 4,
    min_score: float = 0.25,
) -> list[dict]:
    """Retrieve the most relevant chunks for a user query."""
    # The same embedding model used at build time must encode the query.
    query_embedding = create_embeddings([query], embedding_model, show_progress_bar=False)
    scores, indexes = search_index(index, query_embedding, top_k)

    results = []
    for score, chunk_index in zip(scores, indexes):
        if chunk_index == -1 or float(score) < min_score:
            continue

        chunk = chunks[int(chunk_index)]
        results.append(
            {
                "score": float(score),
                "id": chunk["id"],
                "source": chunk["source"],
                "text": chunk["text"],
            }
        )

    return results


def format_context(results: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved chunks for a generation prompt."""
    context_blocks = []
    used_chars = 0

    for result in results:
        block = result["text"].strip()
        if used_chars + len(block) > max_chars:
            break
        context_blocks.append(block)
        used_chars += len(block)

    return "\n\n---\n\n".join(context_blocks)
