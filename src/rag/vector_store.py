from pathlib import Path

import numpy as np


def save_faiss_index(embeddings: np.ndarray, output_path: Path) -> None:
    """Save normalized embeddings in a FAISS index."""
    try:
        import faiss
    except ImportError as error:
        raise ImportError("Install faiss-cpu to save and search the vector index.") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dimension = embeddings.shape[1]
    # Embeddings are normalized, so inner product behaves like cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(output_path))


def load_faiss_index(index_path: Path):
    """Load a FAISS index from disk."""
    try:
        import faiss
    except ImportError as error:
        raise ImportError("Install faiss-cpu to save and search the vector index.") from error

    return faiss.read_index(str(index_path))


def search_index(index, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Search the most similar chunks."""
    scores, indexes = index.search(query_embedding.astype("float32"), top_k)
    return scores[0], indexes[0]
