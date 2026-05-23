# 🌎 RAG Chatbot — Latin America Shares

Chatbot with a RAG architecture (Retrieval-Augmented Generation) for Latin America Shares.  
It uses the chunking and embeddings logic from the professor’s project and the visual interface developed by the team.

---

## Project Structure


proyecto_final/
├── app.py # Flask server
├── requirements.txt
├── ejecutar_linux_mac.sh
├── ejecutar_windows.bat
├── data/
│ ├── raw/ # Source documents (.txt, .pdf, .docx)
│ └── processed/
│ ├── chunks.jsonl # Generated chunks
│ └── embeddings.npy # NumPy embeddings
├── indexes/
│ └── faiss.index # FAISS index
├── scripts/
│ └── build_knowledge_base.py # Generates chunks + embeddings + FAISS
├── src/rag/
│ ├── text_cleaner.py # Text cleaning
│ ├── chunker.py # Semantic chunking (professor’s logic)
│ ├── document_loader.py # Loads .txt, .pdf, .docx files
│ ├── embeddings.py # Multilingual embeddings
│ ├── vector_store.py # FAISS index
│ ├── retriever.py # Semantic search
│ ├── generator.py # Generation with Qwen
│ └── pipeline.py # Full pipeline + anti-hallucination filters
├── static/
│ ├── css/styles.css # Latin America Shares purple color palette
│ └── js/chat.js
└── templates/
└── index.html


---

## How to Run

### Linux / macOS

```bash
chmod +x ejecutar_linux_mac.sh
./ejecutar_linux_mac.sh
Windows
ejecutar_windows.bat

Then open: http://127.0.0.1:5000

Adding Documents

Place your .txt, .pdf, or .docx files inside data/raw/ and run again:

cd scripts
python build_knowledge_base.py --download-model
Features
Semantic chunking with section boundary detection, overlapping, and validation
Multilingual embeddings using paraphrase-multilingual-MiniLM-L12-v2
FAISS index with cosine similarity
Qwen2.5-0.5B-Instruct generation with a strict prompt
Anti-hallucination system: domain filter, ground truth verification, and controlled fallback response
Purple UI based on the visual identity of Latin America Shares
