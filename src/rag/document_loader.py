from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def load_document(path: Path) -> str:
    """Load text from a supported document."""
    extension = path.suffix.lower()

    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    if extension == ".docx":
        return _load_docx(path)

    if extension == ".pdf":
        return _load_pdf(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_documents(input_dir: Path) -> list[dict]:
    """Load every supported document from a folder."""
    documents = []

    for path in sorted(input_dir.rglob("*")):
        # Every returned document keeps its source path for traceability in chunks.
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(
                {
                    "source": str(path),
                    "text": load_document(path),
                }
            )

    return documents


def _load_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as error:
        raise ImportError("Install python-docx to read .docx files.") from error

    document = Document(path)
    # Preserve paragraph boundaries because the chunker depends on them.
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError("Install pypdf to read .pdf files.") from error

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())
