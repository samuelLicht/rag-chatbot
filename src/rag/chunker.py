import re
from dataclasses import dataclass

from rag.text_cleaner import clean_text, split_paragraphs


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    start_paragraph: int
    end_paragraph: int
    word_count: int
    validation_status: str
    validation_notes: list[str]


@dataclass
class ChunkingConfig:
    min_words: int = 35
    target_words: int = 140
    max_words: int = 240
    overlap_sentences: int = 0


def build_chunks(text: str, source: str, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Create semantic chunks that prefer section boundaries over raw size."""
    config = config or ChunkingConfig()
    paragraphs = _prepare_paragraphs(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    start_index = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_words = _count_words(paragraph)
        current_words = _count_words(" ".join(current))
        paragraph_starts_section = _is_section_boundary(paragraph)

        # Start a new chunk at real topic changes, even if the previous block is short.
        boundary_min_words = max(20, config.min_words // 2)
        if current and paragraph_starts_section and current_words >= boundary_min_words:
            chunks.append(_create_chunk(chunks, current, source, start_index, paragraph_index - 1, config))
            current = []
            start_index = paragraph_index
            current_words = _count_words(" ".join(current))

        if current and current_words + paragraph_words > config.max_words:
            chunks.append(_create_chunk(chunks, current, source, start_index, paragraph_index - 1, config))
            current = _build_overlap(current, config.overlap_sentences)
            start_index = max(start_index, paragraph_index - len(current))
            current_words = _count_words(" ".join(current))

        if paragraph_words > config.max_words:
            # Very long paragraphs are split by sentences as a last resort.
            sentence_chunks = _split_long_paragraph(paragraph, config.max_words)
            for sentence_chunk in sentence_chunks:
                candidate = current + [sentence_chunk]
                if _count_words(" ".join(candidate)) >= config.min_words:
                    chunks.append(_create_chunk(chunks, candidate, source, start_index, paragraph_index, config))
                    current = _build_overlap(candidate, config.overlap_sentences)
                    start_index = paragraph_index
                else:
                    current = candidate
            continue

        current.append(paragraph)

        # Prefer cutting near the target size only when the next paragraph changes topic.
        next_paragraph = paragraphs[paragraph_index + 1] if paragraph_index + 1 < len(paragraphs) else ""
        current_words = _count_words(" ".join(current))
        if current_words >= config.target_words and (
            _is_section_boundary(next_paragraph) or current_words >= int(config.max_words * 0.8)
        ):
            chunks.append(_create_chunk(chunks, current, source, start_index, paragraph_index, config))
            current = [] if _is_section_boundary(next_paragraph) else _build_overlap(current, config.overlap_sentences)
            start_index = paragraph_index if current else paragraph_index + 1

    if current:
        current_words = _count_words(" ".join(current))
        if chunks and current_words < config.min_words and chunks[-1].word_count + current_words <= config.max_words:
            chunks[-1] = _merge_last_chunk(chunks[-1], current, config)
        else:
            chunks.append(_create_chunk(chunks, current, source, start_index, len(paragraphs) - 1, config))

    return _merge_weak_chunks(chunks, config)


def _prepare_paragraphs(text: str) -> list[str]:
    paragraphs = split_paragraphs(clean_text(text))
    # The website draft includes instructions for editors; those should not reach the chatbot.
    paragraphs = _drop_editorial_paragraphs(paragraphs)
    paragraphs = _split_structural_paragraphs(paragraphs)
    return _drop_editorial_paragraphs(paragraphs)


def chunk_to_record(chunk: Chunk) -> dict:
    """Convert a chunk to a JSON-serializable dictionary."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "start_paragraph": chunk.start_paragraph,
        "end_paragraph": chunk.end_paragraph,
        "word_count": chunk.word_count,
        "validation_status": chunk.validation_status,
        "validation_notes": chunk.validation_notes,
    }


def _create_chunk(
    chunks: list[Chunk],
    paragraphs: list[str],
    source: str,
    start_paragraph: int,
    end_paragraph: int,
    config: ChunkingConfig,
) -> Chunk:
    text = "\n\n".join(paragraphs).strip()
    notes = _validate_chunk_text(text, config)
    status = "ok" if not notes else "review"

    return Chunk(
        id=f"chunk-{len(chunks) + 1:04d}",
        text=text,
        source=source,
        start_paragraph=start_paragraph,
        end_paragraph=end_paragraph,
        word_count=_count_words(text),
        validation_status=status,
        validation_notes=notes,
    )


def _merge_last_chunk(last_chunk: Chunk, extra_paragraphs: list[str], config: ChunkingConfig) -> Chunk:
    text = "\n\n".join([last_chunk.text, *extra_paragraphs]).strip()
    notes = _validate_chunk_text(text, config)

    return Chunk(
        id=last_chunk.id,
        text=text,
        source=last_chunk.source,
        start_paragraph=last_chunk.start_paragraph,
        end_paragraph=last_chunk.end_paragraph + len(extra_paragraphs),
        word_count=_count_words(text),
        validation_status="ok" if not notes else "review",
        validation_notes=notes,
    )


def _merge_weak_chunks(chunks: list[Chunk], config: ChunkingConfig) -> list[Chunk]:
    if not chunks:
        return []

    # A short heading or list intro is merged forward so the final chunk reads complete.
    merged_chunks = chunks[:]
    result: list[Chunk] = []
    index = 0

    while index < len(merged_chunks):
        chunk = merged_chunks[index]
        next_index = index + 1

        if next_index < len(merged_chunks) and _should_merge_forward(chunk, merged_chunks[next_index], config):
            merged_chunks[next_index] = _merge_two_chunks(chunk, merged_chunks[next_index], config)
            index += 1
            continue

        result.append(chunk)
        index += 1

    return result


def _should_merge_forward(chunk: Chunk, next_chunk: Chunk, config: ChunkingConfig) -> bool:
    combined_words = chunk.word_count + next_chunk.word_count
    if combined_words > config.max_words:
        return False

    return chunk.word_count < config.min_words or bool(re.search(r"[,;:]$", chunk.text))


def _merge_two_chunks(first: Chunk, second: Chunk, config: ChunkingConfig) -> Chunk:
    text = "\n\n".join([first.text, second.text]).strip()
    notes = _validate_chunk_text(text, config)

    return Chunk(
        id=first.id,
        text=text,
        source=first.source,
        start_paragraph=first.start_paragraph,
        end_paragraph=second.end_paragraph,
        word_count=_count_words(text),
        validation_status="ok" if not notes else "review",
        validation_notes=notes,
    )


def _build_overlap(paragraphs: list[str], overlap_sentences: int) -> list[str]:
    if overlap_sentences <= 0 or not paragraphs:
        return []

    sentences = _split_sentences(paragraphs[-1])
    overlap = " ".join(sentences[-overlap_sentences:]).strip()
    return [overlap] if overlap else []


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    sentence_chunks = []
    current_sentences = []

    for sentence in _split_sentences(paragraph):
        candidate = " ".join([*current_sentences, sentence])
        if current_sentences and _count_words(candidate) > max_words:
            sentence_chunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_sentences:
        sentence_chunks.append(" ".join(current_sentences).strip())

    return sentence_chunks


def _split_structural_paragraphs(paragraphs: list[str]) -> list[str]:
    expanded: list[str] = []

    for paragraph in paragraphs:
        # PDF extraction can glue section titles into one paragraph; split them back out.
        parts = [part.strip() for part in re.split(_STRUCTURAL_SPLIT_PATTERN, paragraph) if part.strip()]
        for part in parts or [paragraph]:
            if re.match(r"^\d+\.?$", part) and expanded:
                expanded[-1] = f"{expanded[-1]} {part}"
            elif expanded and re.search(r"\b\d+\.?$", expanded[-1]) and _is_section_boundary(f"{expanded[-1].split()[-1]} {part}"):
                number = expanded[-1].split()[-1]
                expanded[-1] = expanded[-1].removesuffix(number).strip()
                expanded.append(f"{number} {part}".strip())
            else:
                expanded.append(part)

    return expanded


def _drop_editorial_paragraphs(paragraphs: list[str]) -> list[str]:
    cleaned: list[str] = []

    for paragraph in paragraphs:
        paragraph = _clean_editorial_fragments(paragraph)
        if not paragraph or _is_editorial_paragraph(paragraph):
            continue
        cleaned.append(paragraph)

    return cleaned


def _clean_editorial_fragments(paragraph: str) -> str:
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    paragraph = re.sub(r"\s*Documento preparado para sistemas\b.*$", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"\s*\(crear contenido\)\s*y\s*crear imágenes de nuevas noticias relevantes\.?", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"^Título:\s*", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"^Subtítulo:\s*", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"^CTA\s*\(botón\):\s*", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"^Bot[oó]n\s+(?:clic|click)[^:]*:\s*", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"^Bot[oó]n,?\s*", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"^👉\s*", "", paragraph)
    paragraph = re.sub(r"\s*(?:Y\s+)?lo lleva a\b.*$", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"\s*Hacen clic\b.*$", ".", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"\s*Hablemos clik\b.*$", "Hablemos.", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r",\s*Hablemos\.", ". Hablemos.", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r"\s*\(ACTUALIZAR FOTOS\)\s*$", "", paragraph, flags=re.IGNORECASE)
    paragraph = re.sub(r":{5,}", "", paragraph)
    paragraph = re.sub(r"\.{2,}", ".", paragraph)
    return paragraph.strip(" -")


def _is_editorial_paragraph(paragraph: str) -> bool:
    normalized = paragraph.strip()
    upper = normalized.upper()

    if not normalized:
        return True

    exact_noise = {
        "PAGINA WEB COLOMBIA COMPARTE.",
        "BANNERS- CARRUSEL",
        "VISUAL RECOMENDADO:",
        "CONTACTO",
        "AULA",
        "SECCIÓN INTERATIVA: PENDIENTE",
        "NOTAS IMPORTANTES A TENER EN CUENTA EN LA PAGINA:",
        "PROGRAMAS.",
        "ELIMINAR BLOG.",
        "VER EQUIPO",
        "EDIFICA",
    }
    if upper in exact_noise:
        return True

    # Secciones de meta-instrucciones: el section title y su contenido quedan en
    # un solo párrafo tras el split por líneas en blanco. Se filtra por prefijo.
    meta_section_prefixes = (
        "TONO DEL CHATBOT",
        "REGLAS DEL CHATBOT",
        "DONACIONES",
        "RESPUESTA CUANDO NO HAY INFORMA",
        "MENSAJE DE BIENVENIDA",
        "FRASES CLAVE",
    )
    if any(upper.startswith(prefix) for prefix in meta_section_prefixes):
        return True

    noise_patterns = [
        r"^Eliminar\b",
        r"^Cambiar\b",
        r"^Dejar\b",
        r"^SECCIÓN\s+\d+\b",
        r"^ACTUALIDAD\b.*\b(eliminar|ocultar)\b",
        r"^Foto\b",
        r"^Tonos\b",
        r"^Imagen\b",
        r"^Sello\b",
        r"^Crear contenido\b",
        r"^ANEXAR\b",
        r"^Desarrollar contenido\b",
        r"^Conocer\b",
        r"^Conoce\b",
        r"^click\s+.*Conoce\b",
        r"^Quiero\b",
        r"^Ver más\b",
        r"^Aquí debe venir\b",
        r"^En la sub-?sesión actual dice\b",
        r"^\d+\.\s*En la sub-?sesión actual dice\b",
        r".*\(esta aparece tal como.*",
        r"^\d+\.EN EL formulario\b",
        r"^\d+\.\s*Direccionar\b",
        r"^\d+\.\s*En el pie de pagina\b",
        r"^Documento preparado para sistemas\b",
        # Instrucciones de tono/reglas del chatbot embebidas en el doc fuente
        r"^El chatbot debe\b",
        r"^No debe mencionar\b",
        r"^No debe entregar\b",
        r"^Cuando el usuario pregunte\b",
        r"^Si alguien pregunta\b",
        r"^Si el usuario pregunta\b",
        r"^Para recibir informaci",
        r"^Gracias por tu pregunta\.",
    ]
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in noise_patterns)


_STRUCTURAL_SPLIT_PATTERN = re.compile(
    r"\s+(?=(?:"
    r"(?:[1-9]\d?\.\s*(?:Contexto|Perfil|Pilares|Impacto|Formas|Información|EDIFICA|TOP|Empresas|En\s+la))|"
    r"(?:A\s+(?:Personas|Emprendedores)\b)|"
    r"(?:[💼🚀🔸]\s*)|"
    r"(?:SECCIÓN\s+\d+)|"
    r"(?:SLIDE\s+\d+)|"
    r"(?:BANNERS[-\s])|"
    r"(?:NUESTRA\s+HISTORIA)|"
    r"(?:NUESTRO\s+IMPACTO)|"
    r"(?:NUESTRAS\s+ACCIONES)|"
    r"(?:SOBRE\s+NOSOTROS)|"
    r"(?:PROGRAMAS\.)|"
    r"(?:NOTAS\s+IMPORTANTES)|"
    r"(?:Información\s+de\s+Contacto)"
    r"))"
)


def _is_section_boundary(paragraph: str) -> bool:
    if not paragraph:
        return False

    normalized = paragraph.strip()
    upper = normalized.upper()

    if re.match(
        r"^[1-9]\d?\.\s*(Contexto|Perfil|Pilares|Impacto|Formas|Información|EDIFICA|TOP|Empresas|En\s+la)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(r"^(SECCIÓN\s+\d+|SLIDE\s+\d+|BANNERS[-\s]|PROGRAMAS\.|CONTACTO$|AULA$)", upper):
        return True

    # Cada línea principal de Latinoamérica Comparte merece su propio chunk
    # para que el retriever pueda distinguirlas con precisión.
    if re.match(r"^COMPARTE\s+(ACADEMIA|LIDERAZGO|TALENTO)\b", upper):
        return True

    if re.match(r"^(DESKUBRE|ESTRUCTURA)\s*\(", upper):
        return True

    if re.match(r"^[💼🚀🔸]\s*", normalized):
        return True

    if re.match(r"^A\s+(Personas|Emprendedores)\b", normalized, flags=re.IGNORECASE):
        return True

    section_titles = {
        "A QUIÉNES ACOMPAÑAMOS",
        "EMPRESAS COMPROMETIDAS CON SU GENTE",
        "ORGANIZACIONES COMPROMETIDAS CON EL BIENESTAR",
        "NODUS | LIDERAZGO CON MENTALIDAD EMPRENDEDORA",
        "NODUS | LIDERAZGO CON PENSAMIENTO EMPRENDEDOR",
        "NUESTRA HISTORIA",
        "NUESTRO IMPACTO",
        "NUESTRAS ACCIONES",
        "SOBRE NOSOTROS – QUIENES SOMOS.",
        "EDIFICA-",
        "ACTUALIDAD – DEJAR SOLO NOTICIAS- ELIMINAR O OCULTAR LA PESTAÑA DE BLOGS.",
        "NOTAS IMPORTANTES A TENER EN CUENTA EN LA PAGINA:",
    }
    if upper in section_titles:
        return True

    return False


def _validate_chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    notes = []
    word_count = _count_words(text)

    if word_count > config.max_words:
        notes.append(f"Chunk exceeds max_words ({word_count}>{config.max_words}).")

    if word_count < config.min_words:
        notes.append(f"Chunk is below min_words ({word_count}<{config.min_words}).")

    if re.search(r"[,;:]$", text):
        notes.append("Chunk ends with punctuation that suggests an incomplete idea.")

    if re.search(r"\b(y|o|de|del|la|el|los|las|que|para|con|por)$", text, flags=re.IGNORECASE):
        notes.append("Chunk ends with a connector or article.")

    return notes


def _split_sentences(text: str) -> list[str]:
    # Simple Spanish-friendly sentence split. It keeps punctuation at sentence end.
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\wáéíóúÁÉÍÓÚñÑüÜ]+\b", text))
