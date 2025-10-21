from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer


app = Flask(__name__)


# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'invalid or missing JSON body'}), 400

    texts = data.get('inputs', [])
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, (list, tuple)):
        return jsonify({'error': '"inputs" must be a string or a list of strings'}), 400

    if len(texts) == 0:
        return jsonify({'embeddings': []}), 200

    try:
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'embeddings': embeddings})


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)