"""
rank/rank_sections.py
─────────────────────
Return the top-k highest-scoring UNIQUE sections
(key = doc + page + heading) for any query.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import pickle, faiss, logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Hardcoded Docker paths
ROOT        = Path("/app")
INDEX_PATH  = ROOT / "vector_store/index.faiss"
CHUNKS_PATH = ROOT / "vector_store/chunks.pkl"
EMBED_MODEL = "intfloat/e5-base-v2"

# Load components with error handling
try:
    logging.info("Loading FAISS index from: %s", INDEX_PATH)
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    _index = faiss.read_index(str(INDEX_PATH))
    
    logging.info("Loading chunks from: %s", CHUNKS_PATH)
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")
    _chunks = pickle.load(open(CHUNKS_PATH, "rb"))
    
    logging.info("Loading sentence transformer: %s", EMBED_MODEL)
    _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    
    if _index.d != _embedder.get_sentence_embedding_dimension():
        raise ValueError("Vector-store dim ≠ embedder dim — rebuild needed")
        
except Exception as e:
    logging.error("Failed to load ranking components: %s", e)
    raise

def rank_sections(query: str, k: int = 5) -> List[Dict]:
    """
    Top-k UNIQUE sections.
    """
    qv = _embedder.encode([f"query: {query}"],
                          normalize_embeddings=True).astype("float32")
    D, I = _index.search(qv, k * 3)          # heavy over-fetch

    seen_keys: set[Tuple] = set()
    out:  List[Dict] = []

    for row, idx in enumerate(I[0]):
        ch = _chunks[idx]
        key = (ch["doc"], ch["page"], ch["heading"])
        if key in seen_keys:
            continue

        seen_keys.add(key)
        out.append({
            "doc":     ch["doc"],
            "page":    ch["page"],
            "heading": ch["heading"],
            "score":   float(D[0][row]),
            "text":    ch["text"]
        })
        if len(out) == k:
            break
    return out