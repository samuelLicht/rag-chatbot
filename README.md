# 🌎 Chatbot RAG — Latinoamérica Comparte

Chatbot con arquitectura RAG (Retrieval-Augmented Generation) para Latinoamérica Comparte.  
Usa la lógica de chunking y embeddings del proyecto del profesor y la interfaz visual del equipo de desarrollo.

---

## Estructura del proyecto

```
proyecto_final/
├── app.py                        # Servidor Flask
├── requirements.txt
├── ejecutar_linux_mac.sh
├── ejecutar_windows.bat
├── data/
│   ├── raw/                      # Documentos fuente (.txt, .pdf, .docx)
│   └── processed/
│       ├── chunks.jsonl          # Chunks generados
│       └── embeddings.npy        # Embeddings numpy
├── indexes/
│   └── faiss.index               # Índice FAISS
├── scripts/
│   └── build_knowledge_base.py  # Genera chunks + embeddings + FAISS
├── src/rag/
│   ├── text_cleaner.py           # Limpieza de texto
│   ├── chunker.py                # Chunking semántico (lógica del profe)
│   ├── document_loader.py        # Carga de .txt, .pdf, .docx
│   ├── embeddings.py             # Embeddings multilingüe
│   ├── vector_store.py           # FAISS index
│   ├── retriever.py              # Búsqueda semántica
│   ├── generator.py              # Generación con Qwen
│   └── pipeline.py               # Pipeline completo + filtros anti-alucinación
├── static/
│   ├── css/styles.css            # Paleta morada Latinoamérica Comparte
│   └── js/chat.js
└── templates/
    └── index.html
```

---

## Cómo ejecutar

### Linux / macOS
```bash
chmod +x ejecutar_linux_mac.sh
./ejecutar_linux_mac.sh
```

### Windows
```
ejecutar_windows.bat
```

Luego abre: **http://127.0.0.1:5000**

---

## Agregar documentos

Coloca tus archivos `.txt`, `.pdf` o `.docx` en `data/raw/` y vuelve a correr:

```bash
cd scripts
python build_knowledge_base.py --download-model
```

---

## Características

- **Chunking semántico** con detección de límites de sección, overlapping y validación
- **Embeddings multilingüe** con `paraphrase-multilingual-MiniLM-L12-v2`
- **Índice FAISS** con similaridad coseno
- **Generación Qwen2.5-0.5B-Instruct** con prompt estricto
- **Anti-alucinación**: filtro de dominio, verificación de ground truth, fallback controlado
- **UI morada** con identidad visual de Latinoamérica Comparte
