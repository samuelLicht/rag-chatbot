from __future__ import annotations
import re

DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_ANSWER = (
    "En este momento no tengo información específica sobre ese tema. "
    "Te recomiendo comunicarte directamente con el equipo de Latinoamérica Comparte "
    "a través de la sección de contacto o escribir a comunicaciones@colombiacomparte.com "
    "para recibir orientación personalizada."
)


def load_generation_model(model_name: str = DEFAULT_GENERATION_MODEL, local_files_only: bool = True):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Instala torch y transformers para generar respuestas.") from error

    model_path = _resolve_model_path(model_name, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, local_files_only=local_files_only)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=local_files_only)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_answer(
    query: str,
    context: str,
    tokenizer,
    model,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    num_questions: int = 1,
    history: list | None = None,
) -> str:
    if not context.strip():
        return FALLBACK_ANSWER

    # Limitar historial a los últimos 3 turnos para no saturar la ventana de contexto
    trimmed_history = (history or [])[-6:]
    prompt = _build_messages(query, context, num_questions=num_questions, history=trimmed_history)
    if num_questions >= 2:
        max_new_tokens = min(max_new_tokens * num_questions, 900)
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _clean_answer(answer)


def _build_messages(query: str, context: str, num_questions: int = 1, history: list | None = None) -> list[dict]:
    if num_questions >= 2:
        length_instruction = (
            f"La pregunta contiene {num_questions} partes. "
            "Responde CADA parte en una o dos frases, en el mismo orden en que fueron preguntadas."
        )
    else:
        length_instruction = "Contesta de forma completa usando términos del contexto. Si hay varios ítems, explica CADA uno sin omitir ninguno."

    system_prompt = (
        "Eres el asistente virtual oficial de Latinoamérica Comparte. "
        "Responde ÚNICAMENTE usando la información del CONTEXTO proporcionado. "
        "\n\n"
        "JERARQUÍA DE PROGRAMAS (muy importante):\n"
        "- Las 3 LÍNEAS PRINCIPALES son: Comparte Academia, Comparte Liderazgo y Comparte Talento.\n"
        "- DESKUBRE y ESTRUCTURA son sub-programas que pertenecen DENTRO de Comparte Academia, NO son líneas principales.\n"
        "- Comparte Talento se enfoca en speakers, conferencistas y eventos corporativos (antes llamado Top Speakers).\n"
        "- Comparte Liderazgo se enfoca en liderazgo, cultura organizacional y desarrollo humano en empresas.\n"
        "\n"
        "REGLAS:\n"
        "- Cuando el usuario pregunte por los programas o líneas, SIEMPRE menciona las 3 líneas principales: "
        "Comparte Academia, Comparte Liderazgo, Comparte Talento.\n"
        "- NUNCA listes DESKUBRE o ESTRUCTURA como si fueran líneas principales.\n"
        "- NUNCA menciones EDIFICA ni ninguna variación de ese nombre.\n"
        "- NUNCA inventes datos, cifras o nombres que no estén en el contexto.\n"
        "- NUNCA entregues información sobre donaciones o cuentas bancarias.\n"
        f"- Si el contexto no contiene la respuesta, di exactamente: '{FALLBACK_ANSWER}'\n"
        "- Responde en español con tono cálido y profesional.\n"
        "- " + length_instruction
    )
    user_prompt = (
        f"Contexto recuperado:\n{context}\n\n"
        f"Pregunta del usuario:\n{query}\n\n"
        "Respuesta (responde SOLO lo que se preguntó, usando el contexto):"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Inyectar turnos previos de la conversación para dar contexto al LLM
    for turn in (history or []):
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def _clean_answer(answer: str) -> str:
    answer = answer.strip()
    # Eliminar prefijos de rol
    answer = re.sub(r"^(Respuesta:|Asistente:|assistant\n)", "", answer, flags=re.IGNORECASE)
    # Eliminar marcadores de contexto que el modelo haya repetido
    stop_markers = ["\nCONTEXTO", "\nPREGUNTA", "\nUsuario:", "\nAsistente:"]
    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    # Limpiar tokens especiales
    for token in ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
        answer = answer.replace(token, "")
    # Eliminar encabezados markdown y prefijos en MAYÚSCULAS que el modelo copia del contexto
    answer = re.sub(r"^#{1,6}\s+.+$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"^[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{1,})*\s+(?=[A-Za-záéíóúñ])", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    # Corrección de nombres de programas propios que el modelo suele malescribir
    name_corrections = [
        (r'\bDeskub[eo][rs]?\b', 'DESKUBRE'),
        (r'\bdeskub[eo][rs]?\b', 'DESKUBRE'),
        (r'\bEstructur[ae]\b', 'ESTRUCTURA'),
        (r'\bEdific[ae]\b', 'EDIFICA'),
    ]
    for pattern, replacement in name_corrections:
        answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)

    # Frases problemáticas que indican alucinación
    bad_phrases = [
        "según el contexto", "basándome en el contexto",
        "de acuerdo con el contexto", "la información proporcionada",
        "en el contexto", "con base en el contexto",
    ]
    for phrase in bad_phrases:
        if phrase.lower() in answer.lower():
            idx = answer.lower().find(phrase.lower())
            answer = answer[:idx].strip()
    # Eliminar líneas que son solo número+punto sin texto (ítems incompletos)
    answer = re.sub(r"^\s*\d+\.\s*$", "", answer, flags=re.MULTILINE).strip()
    # Asegurar que termina con puntuación
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    complete = [s.strip() for s in sentences if s.strip() and s.strip()[-1] in ".!?"]
    if complete:
        answer = " ".join(complete[:12]).strip()
    elif answer and answer[-1] not in ".!?":
        answer += "."
    return answer.strip() or FALLBACK_ANSWER


def _resolve_model_path(model_name: str, local_files_only: bool) -> str:
    if "/" not in model_name:
        return model_name
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name
    return snapshot_download(repo_id=model_name, local_files_only=local_files_only)
