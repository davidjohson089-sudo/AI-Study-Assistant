from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("WIX")

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# ---------------- OCR (Image -> Text) ----------------
@app.route("/ocr", methods=["POST"])
def ocr():
    image_base64 = request.json.get("image")

    response = requests.post(
        "https://api-inference.huggingface.co/models/microsoft/trocr-base-printed",
        headers=headers,
        json={"inputs": image_base64}
    )

    return jsonify(response.json())

# ---------------- TEXT SUMMARY ----------------
@app.route("/summary", methods=["POST"])
def summary():
    text = request.json.get("text")

    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers=headers,
        json={"inputs": text}
    )

    return jsonify(response.json())

# ---------------- QUESTION ANSWERING ----------------
@app.route("/qa", methods=["POST"])
def qa():
    data = request.json
    context = data.get("context")
    question = data.get("question")

    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2",
        headers=headers,
        json=payload
    )

    return jsonify(response.json())

# ---------------- QUIZ (EASY / HARD) ----------------
@app.route("/quiz", methods=["POST"])
def quiz():
    text = request.json.get("text")
    level = request.json.get("level")  # easy or hard

    prompt = f"Generate {level} level questions and answers from this text:\n{text}"

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json={"inputs": prompt}
    )

    return jsonify(response.json())

# ---------------- FLASHCARDS ----------------
@app.route("/flashcards", methods=["POST"])
def flashcards():
    text = request.json.get("text")

    prompt = f"Create flashcards in Q&A format from this text:\n{text}"

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json={"inputs": prompt}
    )

    return jsonify(response.json())

# ---------------- TRANSLATION ----------------
@app.route("/translate", methods=["POST"])
def translate():
    text = request.json.get("text")
    lang = request.json.get("lang")

    prompt = f"Translate this into {lang}: {text}"

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json={"inputs": prompt}
    )

    return jsonify(response.json())

@app.route("/")
def home():
    return "AI Study API is running!"

if __name__ == "__main__":
    app.run(debug=True)

