@echo off
echo Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate.bat

echo Instalando dependencias...
pip install -r requirements.txt

echo Construyendo base de conocimiento...
cd scripts
python build_knowledge_base.py --download-model
cd ..

echo Iniciando servidor Flask...
python app.py
