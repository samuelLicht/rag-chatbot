import re


def clean_text(text: str) -> str:
    """Normalize text while keeping paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # PDF bullets sometimes arrive as repeated loose symbols; remove that noise early.
    text = re.sub(r"[•·]\s*(?:[•·]\s*)+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split text into clean paragraphs."""
    paragraphs = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs
