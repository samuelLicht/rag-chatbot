import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from flask import Flask, jsonify, render_template, request

try:
    from rag.pipeline import chatbot_response
    RAG_READY = True
except Exception as e:
    RAG_READY = False
    print(f"❌ Error cargando pipeline RAG: {e}")

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"answer": "Por favor escribe una pregunta sobre Latinoamérica Comparte."}), 400

        if not RAG_READY:
            return jsonify({"answer": "El motor RAG no está disponible. Revisa la consola."}), 500

        history = data.get("history", [])
        if not isinstance(history, list):
            history = []
        answer = chatbot_response(message, history=history)
        return jsonify({"answer": answer})

    except Exception as error:
        print(f"❌ Error en /chat: {error}")
        return jsonify({"answer": "Ocurrió un error procesando tu pregunta."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
