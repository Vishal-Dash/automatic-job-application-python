# embedding_service.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/embed', methods=['POST'])
def embed():
    payload = request.get_json(force=True)
    # Accept either {"text":"..."} or {"inputs":["a","b"]}
    texts = payload.get("inputs") or payload.get("texts") or payload.get("text")
    if texts is None:
        return jsonify({"error":"no input"}), 400
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    return jsonify({"embeddings": embeddings})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    # Note: models will be downloaded on first run
    app.run(host="0.0.0.0", port=8000)
