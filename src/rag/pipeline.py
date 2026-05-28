from __future__ import annotations
import re
import unicodedata
from pathlib import Path

from rag.embeddings import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from rag.generator import DEFAULT_GENERATION_MODEL, FALLBACK_ANSWER, generate_answer, load_generation_model
from rag.retriever import format_context, load_chunks, retrieve_context
from rag.vector_store import load_faiss_index

# ─── Constantes de dominio ────────────────────────────────────────────────────
CONTACT_INFO = (
    "Puedes contactar a Latinoamérica Comparte a los teléfonos (+57) 321 230 2138 y "
    "(+57) 316 467 3087. Correo general: comunicaciones@colombiacomparte.com. "
    "Información empresarial: Eduardodelcastillo@colombiacomparte.com. "
    "Sección de contacto: https://colombiacomparte.com/contacto/"
)
DONATION_MSG = (
    "En este momento no cuento con información habilitada sobre procesos de donación. "
    "Te recomiendo comunicarte directamente con el equipo de Latinoamérica Comparte."
)
SELF_INTRO = (
    "Soy un asistente virtual disponible para ti en todo momento. "
    "Estoy aquí para ayudarte a conocer todo sobre Latinoamérica Comparte: "
    "sus líneas como Comparte Academia, Comparte Liderazgo y Comparte Talento, "
    "su historia, impacto y cómo pueden acompañarte en tu proceso de transformación "
    "personal o empresarial. Puedes preguntarme lo que necesites."
)

DOMAIN_KEYWORDS = [
    "latinoamerica comparte", "colombia comparte", "comparte academia",
    "comparte liderazgo", "comparte talento", "edifica", "nodus", "top speakers",
    "emprendimiento", "emprendedor", "bienestar", "liderazgo", "transformacion",
    "familias", "personas", "colaboradores", "fundadores", "historia", "impacto",
    "formacion", "capacitacion", "conferencias", "talleres", "coaching",
    "talento humano", "cultura organizacional", "mision", "vision", "valores",
    "paises", "latinoamerica", "ecuador", "chile", "argentina", "red regional",
    "que hacen", "que es", "quienes son", "como funciona", "para que sirve",
    "programas", "lineas", "acompanamiento",
    "testimonios", "testimonio", "experiencias", "youtube", "canal", "video", "videos",
    "redes", "redes sociales", "instagram", "facebook", "tiktok", "linkedin",
    "dinero consciente", "aula", "solicitudes", "dian",
]
CONTACT_KEYWORDS = [
    "contacto", "contactar", "telefono", "numero", "correo", "comunicar", "llamar",
    "como los contacto", "datos de contacto", "escribirles", "hablar con ellos",
    "encontrarlos", "ubicarlos", "donde estan", "sede",
]
DONATION_KEYWORDS = [
    "donar", "donacion", "donaciones", "cuenta bancaria", "transferencia",
    "apoyo economico", "contribucion", "quiero apoyar economicamente",
]
GREETING_KEYWORDS = [
    "hola", "buenos dias", "buenas tardes", "buenas noches", "buenas",
    "como estas", "saludos", "hi", "hello",
]
SELF_INTRO_KEYWORDS = [
    "que haces", "para que sirves", "quien eres", "que eres",
    "como me ayudas", "cual es tu funcion", "que hace el chat",
]
YOUTUBE_MSG = (
    "El canal oficial de YouTube de Latinoamérica Comparte es: "
    "https://www.youtube.com/@colombiacomparte. "
    "Allí encontrarás testimonios, experiencias y contenido de transformación humana."
)
SOCIAL_MSG = (
    "Latinoamérica Comparte tiene presencia en Facebook, Instagram, TikTok, LinkedIn y YouTube. "
    "El canal oficial de YouTube es: https://www.youtube.com/@colombiacomparte"
)
YOUTUBE_KEYWORDS = [
    "canal de youtube", "canal youtube", "youtube", "canal oficial",
]
SOCIAL_KEYWORDS = [
    "redes sociales", "instagram", "facebook", "tiktok", "linkedin",
    "redes de latinoamerica", "redes de la organizacion",
]
SENSITIVE_KEYWORDS = [
    "precio", "costo", "cuesta", "tarifa", "horario", "cupo",
    "fecha exacta", "certificado", "nit", "contrasena", "sede fisica",
]

# Indicadores de que la pregunta es un follow-up que depende del contexto previo
FOLLOWUP_INDICATORS = [
    "cada uno", "cada una", "ellos", "ellas", "eso", "esto", "ese", "esa",
    "dime mas", "dime más", "cuentame mas", "cuéntame más", "ampliar",
    "mas sobre", "más sobre", "y los", "y las", "y el", "y la",
    "en que se diferencian", "en qué se diferencian", "diferencia entre",
]

# Expansiones de query: cuando la pregunta contiene alguna de las palabras clave,
# se anexan términos específicos al embedding de búsqueda para mejorar el recall
# sin modificar la pregunta original que llega al generador.
QUERY_EXPANSIONS: list[tuple[list[str], str]] = [
    (
        ["programa", "programas", "lineas", "linea de accion", "lineas de accion",
         "que ofrecen", "que tienen", "que hacen", "describe", "describeme",
         "cuales son", "cada uno"],
        "Comparte Academia Comparte Liderazgo Comparte Talento tres lineas principales",
    ),
    (
        ["academia", "formacion", "aprendizaje", "capacitacion", "cursos", "deskubre", "estructura"],
        "Comparte Academia DESKUBRE ESTRUCTURA emprendimiento",
    ),
    (
        ["liderazgo", "cultura organizacional", "equipos", "directivos", "empresa"],
        "Comparte Liderazgo liderazgo consciente cultura organizacional",
    ),
    (
        ["talento", "speakers", "conferencistas", "conferencia", "eventos corporativos"],
        "Comparte Talento Top Speakers conferencistas artistas",
    ),
    (
        ["fundadores", "historia", "origen", "como nacio", "quienes fundaron"],
        "Carolina Ruiz Eduardo Del Castillo fundadores historia Colombia",
    ),
    (
        ["impacto", "cuantas personas", "cuantas empresas", "mentores"],
        "impacto 1200 personas 70 empresas 65 mentores",
    ),
]


def _expand_query(query: str) -> str:
    """Agrega términos específicos al query para mejorar el recall del retriever."""
    q = _normalize(query)
    expansions = []
    for keywords, terms in QUERY_EXPANSIONS:
        if any(_normalize(kw) in q for kw in keywords):
            expansions.append(terms)
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def _is_followup(query: str, history: list) -> bool:
    """Detecta si la pregunta es una continuación de una conversación previa."""
    if not history:
        return False
    q = _normalize(query)
    words = q.split()
    if len(words) <= 8 and any(_normalize(fi) in q for fi in FOLLOWUP_INDICATORS):
        return True
    return False


def _get_history_context(history: list) -> str:
    """Extrae el texto de los últimos mensajes del usuario del historial."""
    user_msgs = [m.get("content", "") for m in history if m.get("role") == "user"]
    return " ".join(user_msgs[-2:])  # últimas 2 preguntas del usuario


def _history_has_domain(history: list) -> bool:
    """Verifica si algún mensaje reciente del usuario tocó temas del dominio."""
    for msg in history[-6:]:
        if msg.get("role") == "user" and _matches(msg.get("content", ""), DOMAIN_KEYWORDS):
            return True
    return False


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^a-z0-9@./:+ -]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _matches(query: str, keywords: list[str]) -> bool:
    q = _normalize(query)
    return any(_normalize(kw) in q for kw in keywords)


def _is_grounded(answer: str, context: str) -> bool:
    stop = {"que", "es", "son", "la", "el", "de", "del", "en", "por",
            "para", "con", "como", "su", "sus", "a", "y", "o", "no"}
    a_words = [w for w in _normalize(answer).split() if len(w) > 3 and w not in stop]
    if not a_words:
        return True
    c_words = set(_normalize(context).split())
    overlap = sum(1 for w in a_words if w in c_words)
    return (overlap / len(a_words)) >= 0.30


def _is_echo(query: str, answer: str) -> bool:
    if not answer:
        return True
    nq, na = _normalize(query), _normalize(answer)
    return na in (nq, nq + ".")


def _split_questions(query: str) -> list[str]:
    """Divide una consulta compuesta en sub-preguntas individuales."""
    # Partir por ? (preguntas con signos de interrogación)
    parts = re.split(r"\?", query)
    questions = []
    for part in parts:
        part = re.sub(r"^[\s,;¿]+", "", part)
        part = re.sub(r"^(y\s+|también\s+|tambien\s+|además\s+|ademas\s+)", "", part, flags=re.IGNORECASE).strip()
        if len(part) > 5:
            questions.append(part)

    if len(questions) >= 2:
        return questions

    # Sin signos de ?, detectar conectores fuertes: "y qué/cuál/cómo/cuántos/dónde/quiénes"
    connector_split = re.split(
        r"\s+y\s+(que\s+|qué\s+|cual\s+|cuál\s+|cómo\s+|como\s+|cuantos?\s+|cuántos?\s+|donde\s+|dónde\s+|quienes?\s+|quiénes?\s+|en\s+que\s+|en\s+qué\s+)",
        query,
        flags=re.IGNORECASE,
    )
    if len(connector_split) >= 2:
        rebuilt = []
        i = 0
        while i < len(connector_split):
            chunk = connector_split[i].strip()
            if i + 1 < len(connector_split):
                chunk_next = (connector_split[i + 1] + connector_split[i + 2]).strip() if i + 2 < len(connector_split) else connector_split[i + 1].strip()
                rebuilt.append(chunk)
                rebuilt.append(chunk_next)
                i += 3
            else:
                rebuilt.append(chunk)
                i += 1
        rebuilt = [q for q in rebuilt if len(q) > 5]
        if len(rebuilt) >= 2:
            return rebuilt

    # Detectar "también [pregunta]" o "además [pregunta]" como segunda parte
    also_split = re.split(
        r"\s*[,;]\s*(también\s+|tambien\s+|además\s+|ademas\s+|y\s+también\s+|y\s+tambien\s+)",
        query,
        flags=re.IGNORECASE,
    )
    if len(also_split) >= 2:
        parts2 = [p.strip() for p in also_split if len(p.strip()) > 5 and not re.match(r"^(también|tambien|además|ademas|y\s)", p.strip(), re.IGNORECASE)]
        if len(parts2) >= 2:
            return parts2

    return [query]


# ─── Pipeline principal ───────────────────────────────────────────────────────
class LatamChatbot:
    """Pipeline RAG completo: recuperación + generación con Qwen."""

    def __init__(
        self,
        chunks_path: Path = Path("data/processed/chunks.jsonl"),
        index_path: Path = Path("indexes/faiss.index"),
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        generation_model_name: str = DEFAULT_GENERATION_MODEL,
        local_files_only: bool = True,
    ) -> None:
        self.chunks = load_chunks(chunks_path)
        self.index = load_faiss_index(index_path)
        self.embedding_model = load_embedding_model(
            embedding_model_name, local_files_only=local_files_only)
        self.generation_model_name = generation_model_name
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.generation_model = None

    def ask(self, query: str, history: list | None = None) -> str:
        if not query or not query.strip():
            return "Por favor escribe una pregunta sobre Latinoamérica Comparte."

        history = history or []
        sub_questions = _split_questions(query)

        if len(sub_questions) >= 2:
            answers = []
            for sq in sub_questions:
                ans = self._answer_single(sq.strip(), history=history)
                if ans and ans != FALLBACK_ANSWER:
                    answers.append(ans)
                elif ans == FALLBACK_ANSWER:
                    answers.append(ans)

            real = [a for a in answers if a != FALLBACK_ANSWER]
            if real:
                return "\n\n".join(real)
            return FALLBACK_ANSWER

        return self._answer_single(query, history=history)

    def _answer_single(self, query: str, history: list | None = None) -> str:
        """Responde una sola pregunta simple."""
        history = history or []

        # Atajos rápidos sin RAG
        if _matches(query, DONATION_KEYWORDS):
            return DONATION_MSG
        if _matches(query, CONTACT_KEYWORDS):
            return CONTACT_INFO
        if _matches(query, YOUTUBE_KEYWORDS):
            return YOUTUBE_MSG
        if _matches(query, SOCIAL_KEYWORDS):
            return SOCIAL_MSG
        if _matches(query, SELF_INTRO_KEYWORDS):
            return SELF_INTRO
        if _matches(query, GREETING_KEYWORDS):
            return (
                "¡Hola! Soy el asistente de Latinoamérica Comparte. "
                "Puedo contarte sobre Comparte Academia, Comparte Liderazgo, "
                "Comparte Talento y mucho más. ¿En qué te puedo ayudar?"
            )
        if _matches(query, SENSITIVE_KEYWORDS):
            return FALLBACK_ANSWER

        # Filtro de dominio — se relaja si la pregunta es un follow-up de una
        # conversación que ya tocó el dominio
        in_domain = _matches(query, DOMAIN_KEYWORDS)
        if not in_domain:
            is_followup = _is_followup(query, history) and _history_has_domain(history)
            if not is_followup:
                return FALLBACK_ANSWER

        # Si es un follow-up con query corta, enriquecer la búsqueda con el contexto previo
        retrieval_query = query
        if _is_followup(query, history):
            prior_ctx = _get_history_context(history)
            if prior_ctx:
                retrieval_query = f"{prior_ctx} {query}"

        # Recuperación
        expanded_query = _expand_query(retrieval_query)
        results = retrieve_context(
            expanded_query, self.chunks, self.index, self.embedding_model,
            top_k=7, min_score=0.20,
        )
        context = format_context(results, max_chars=4000)

        if not context:
            return FALLBACK_ANSWER

        # Generación
        self._load_generator()
        answer = generate_answer(
            query, context, self.tokenizer, self.generation_model,
            history=history,
            num_questions=1,
        )

        # Validación anti-alucinación
        if _is_echo(query, answer) or not _is_grounded(answer, context):
            top = results[0]["text"] if results else ""
            top = re.sub(r"^#{1,6}\s+.+$", "", top, flags=re.MULTILINE)
            top = re.sub(r"^[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{1,})*\s+(?=[A-Za-záéíóúñ])", "", top, flags=re.MULTILINE)
            top = re.sub(r"\n{3,}", "\n\n", top).strip()
            sentences = re.split(r"(?<=[.!?])\s+", top)
            clean = [s.strip() for s in sentences if s.strip() and s.strip()[-1] in ".!?"]
            return " ".join(clean[:3]).strip() if clean else FALLBACK_ANSWER

        return answer.strip() if answer and answer.strip() else FALLBACK_ANSWER

    def _load_generator(self) -> None:
        if self.tokenizer is not None and self.generation_model is not None:
            return
        self.tokenizer, self.generation_model = load_generation_model(
            self.generation_model_name, local_files_only=self.local_files_only)


# Instancia singleton para Flask
_chatbot_instance: LatamChatbot | None = None


def get_chatbot() -> LatamChatbot:
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = LatamChatbot()
    return _chatbot_instance


def chatbot_response(query: str, history: list | None = None) -> str:
    return get_chatbot().ask(query, history=history or [])
