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
ROOT          = Path(__file__).parent.parent
INDEX_FILE    = ROOT / "vector_store/index.faiss"
CHUNKS_FILE   = ROOT / "vector_store/chunks.pkl"

EMBED_MODEL   = "intfloat/e5-base-v2"
N_CTX         = 1024
TOP_K_DEF     = 1
TRUNC_WORDS   = 250
GEN_TOKENS    = 128
TEMPERATURE   = 0.0

MAX_SUM_WORDS = 35
MARKER_RE     = re.compile(r"\s?(###|---|you are)", re.I)

GGUF_PATH = next((ROOT / "models").glob("*.gguf"), None)
if GGUF_PATH is None:
    raise FileNotFoundError("No .gguf model found in /models")
# ─────────────────────────────────

index    = faiss.read_index(str(INDEX_FILE))
chunks   = pickle.load(open(CHUNKS_FILE, "rb"))
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

llm = Llama(model_path=str(GGUF_PATH), n_ctx=N_CTX, verbose=False)
logging.info("▶ Using GGUF model: %s", GGUF_PATH.name)

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
    q_vec = embedder.encode([f"query: {query}"],
                            normalize_embeddings=True).astype("float32")
    _, I = index.search(q_vec, k)
    hits = []
    for idx in I[0]:
        ch = chunks[idx].copy()
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

    resp = llm(prompt, max_tokens=GEN_TOKENS, temperature=TEMPERATURE)
    return _clean(resp["choices"][0]["text"])


# CLI
if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(description=textwrap.dedent(
        "Quick RAG demo — returns a 2‑sentence answer."))
    ap.add_argument("query"); ap.add_argument("--persona"); ap.add_argument("--job")
    ap.add_argument("-k", type=int, default=TOP_K_DEF)
    a = ap.parse_args()
    print("\n🟢 Answer:\n", generate_answer(a.query, a.persona, a.job, a.k))