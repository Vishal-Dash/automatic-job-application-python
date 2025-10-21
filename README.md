Install dependencies:

pip install flask sentence-transformers scikit-learn torch

Run with:

python embedding_service.py

This will host an API at http://localhost:8000/embed that accepts POST requests with JSON:

{
  "inputs": ["text1", "text2"]
}

and returns:
