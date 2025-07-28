# src/query_index.py
"""
Quick CLI to interrogate the vector_store with a free-text query.

Example:
    python src/query_index.py "How do I create fillable forms?" -k 8
"""

import os, argparse, pickle, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
VSTORE_DIR  = Path(__file__).resolve().parent.parent / "vector_store"
INDEX_FILE  = VSTORE_DIR / "index.faiss"
CHUNKS_FILE = VSTORE_DIR / "chunks.pkl"

# ─────────────────────────────────────────────────────────────
# Load FAISS + metadata + embedder
# ─────────────────────────────────────────────────────────────
index   = faiss.read_index(str(INDEX_FILE))
chunks  = pickle.load(open(CHUNKS_FILE, "rb"))
embedder = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

# ─────────────────────────────────────────────────────────────
# Search helper
# ─────────────────────────────────────────────────────────────
def search(query: str, k: int = 5):
    q_vec = embedder.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    ).astype("float32")
    D, I = index.search(q_vec, k)
    return [(chunks[idx], float(D[0][j])) for j, idx in enumerate(I[0])]

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ad-hoc semantic search over vector_store")
    ap.add_argument("query", help="Natural-language query")
    ap.add_argument("-k", type=int, default=5, help="Number of hits to show")
    args = ap.parse_args()

    for hit, score in search(args.query, args.k):
        print(f"{score:.3f} | {hit['heading']}  (p.{hit['page']})")
