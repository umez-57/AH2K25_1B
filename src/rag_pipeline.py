"""
Light‑weight RAG
────────────────
• retrieve top‑k chunks from FAISS
• build a persona‑aware prompt
• generate a concise answer (local GGUF via llama‑cpp)
"""

import logging, pickle, faiss, re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ───────── configuration ─────────
# Hardcoded Docker paths
ROOT          = Path("/app")
INDEX_FILE    = ROOT / "vector_store/index.faiss"
CHUNKS_FILE   = ROOT / "vector_store/chunks.pkl"
MODELS_DIR    = ROOT / "models"

EMBED_MODEL   = "intfloat/e5-base-v2"
N_CTX         = 1024
TOP_K_DEF     = 1
TRUNC_WORDS   = 250
GEN_TOKENS    = 128
TEMPERATURE   = 0.0

MAX_SUM_WORDS = 35
MARKER_RE     = re.compile(r"\s?(###|---|you are)", re.I)

# Hardcoded GGUF model path for Docker
GGUF_PATH = MODELS_DIR / "gemma-3-1b-it-Q4_K_M.gguf"
logging.info("ROOT path: %s", ROOT)
logging.info("Models directory: %s", MODELS_DIR)
logging.info("GGUF_PATH: %s", GGUF_PATH)

if not GGUF_PATH.exists():
    logging.error("GGUF model not found at: %s", GGUF_PATH)
    logging.error("Available files in models directory:")
    if MODELS_DIR.exists():
        for file in MODELS_DIR.iterdir():
            logging.error("  - %s", file.name)
    else:
        logging.error("Models directory does not exist: %s", MODELS_DIR)
    raise FileNotFoundError(f"GGUF model not found at {GGUF_PATH}")
# ─────────────────────────────────

# Lazy loading variables
_index = None
_chunks = None
_embedder = None
_llm = None

def _load_components():
    """Lazy load all components when needed."""
    global _index, _chunks, _embedder, _llm
    
    if _index is None:
        logging.info("Loading FAISS index from: %s", INDEX_FILE)
        if not INDEX_FILE.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}")
        _index = faiss.read_index(str(INDEX_FILE))
    
    if _chunks is None:
        logging.info("Loading chunks from: %s", CHUNKS_FILE)
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}")
        _chunks = pickle.load(open(CHUNKS_FILE, "rb"))
    
    if _embedder is None:
        logging.info("Loading sentence transformer: %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    
    if _llm is None:
        logging.info("Loading GGUF model from: %s", GGUF_PATH)
        try:
            _llm = Llama(model_path=str(GGUF_PATH), n_ctx=N_CTX, verbose=False)
            logging.info("▶ Successfully loaded GGUF model: %s", GGUF_PATH.name)
        except Exception as e:
            logging.error("Failed to load GGUF model: %s", e)
            logging.error("Model path: %s", GGUF_PATH)
            logging.error("Model exists: %s", GGUF_PATH.exists())
            logging.error("Model size: %s bytes", GGUF_PATH.stat().st_size if GGUF_PATH.exists() else "N/A")
            raise

# ───────── helpers ─────────
def _clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]+`", " ", text)
    m = MARKER_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    sents = re.split(r"(?<=[.!?]) +", text.strip())
    short = " ".join(sents[:2])
    words = short.split()
    if len(words) > MAX_SUM_WORDS:
        short = " ".join(words[:MAX_SUM_WORDS]) + " …"
    return short.strip()


def retrieve(query: str, k: int = TOP_K_DEF):
    _load_components()
    q_vec = _embedder.encode([f"query: {query}"],
                            normalize_embeddings=True).astype("float32")
    _, I = _index.search(q_vec, k)
    hits = []
    for idx in I[0]:
        ch = _chunks[idx].copy()
        ch["text"] = " ".join(ch["text"].split()[:TRUNC_WORDS])
        hits.append(ch)
    return hits


def generate_answer(query: str, persona: str | None = None,
                    job: str | None = None, k: int = TOP_K_DEF) -> str:

    ctx = retrieve(query, k)
    context = "\n\n".join(
        f"### {c['heading']} (p.{c['page']} in {c['doc']})\n{c['text']}"
        for c in ctx
    )

    if persona:
        sys_role = f"You are a {persona}."
    else:
        sys_role = "You are a helpful assistant."
    if job:
        sys_role += f" The user's task is: {job}"

    prompt = f"""{sys_role}

Use ONLY the CONTEXT below. Reply in **exactly two concise sentences**.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    _load_components()
    resp = _llm(prompt, max_tokens=GEN_TOKENS, temperature=TEMPERATURE)
    return _clean(resp["choices"][0]["text"])


# Expose llm for backward compatibility
def llm(prompt, **kwargs):
    _load_components()
    return _llm(prompt, **kwargs)

# CLI
if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(description=textwrap.dedent(
        "Quick RAG demo — returns a 2‑sentence answer."))
    ap.add_argument("query"); ap.add_argument("--persona"); ap.add_argument("--job")
    ap.add_argument("-k", type=int, default=TOP_K_DEF)
    a = ap.parse_args()
    print("\n🟢 Answer:\n", generate_answer(a.query, a.persona, a.job, a.k))