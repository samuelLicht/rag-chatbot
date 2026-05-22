#!/usr/bin/env bash
# ─── Ejecutar chatbot Latinoamérica Comparte ─────────────────────────────────
set -e

echo "⚙️  Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Instalando dependencias..."
pip install -r requirements.txt

echo "🔨 Construyendo base de conocimiento..."
cd scripts
python build_knowledge_base.py --download-model
cd ..

echo "🚀 Iniciando servidor Flask..."
python app.py
