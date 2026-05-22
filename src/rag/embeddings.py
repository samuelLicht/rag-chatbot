import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL, local_files_only: bool = True):
    """Load a small multilingual Hugging Face embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError("Install sentence-transformers to create embeddings.") from error

    # snapshot_download returns the cached folder when local_files_only=True.
    model_path = _resolve_model_path(model_name, local_files_only)
    return SentenceTransformer(model_path, local_files_only=local_files_only)


def create_embeddings(texts: list[str], model, show_progress_bar: bool = True) -> np.ndarray:
    """Create normalized embeddings for cosine similarity."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return embeddings.astype("float32")


def _resolve_model_path(model_name: str, local_files_only: bool) -> str:
    if "/" not in model_name:
        return model_name

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name

    return snapshot_download(repo_id=model_name, local_files_only=local_files_only)
