# src/build_chunks_and_index.py
"""
Step 2  –  Build heading-anchored chunks, embed them with e5-base-v2,
           and write vector_store/index.faiss + chunks.pkl
"""

import os, json, pickle, re, pathlib, logging
from typing import List, Dict

import numpy as np, pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
EMBED_MODEL_NAME   = "intfloat/e5-base-v2"   # 768-dim, ≈110 MB
MAX_TOKENS         = 512
# Hardcoded Docker paths
INPUT_DIR          = "/app/input"        # PDFs
HEADINGS_DIR       = "/app/output"       # *.headings.json produced by parser
OUT_DIR            = "/app/vector_store"
CHUNKS_PKL         = "chunks.pkl"
FAISS_FILE         = "index.faiss"
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

model     = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
tokenizer = model.tokenizer

# ───────────────────────── helpers ──────────────────────────
def count_tokens(txt: str) -> int:
    return len(tokenizer.encode(txt, add_special_tokens=False))

def split_paragraphs(txt: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]

def chunk_section(heading: str, paras: List[str], doc: str, page: int) -> List[Dict]:
    """Pack heading + subsequent paragraphs into ≤512-token chunks."""
    out, cur, cur_tok = [], [], count_tokens(heading)
    for para in paras:
        p_tok = count_tokens(para)
        if cur and cur_tok + p_tok > MAX_TOKENS:
            out.append({"text": heading + "\n\n" + "\n\n".join(cur),
                        "doc": doc, "page": page, "heading": heading})
            cur, cur_tok = [], count_tokens(heading)
        cur.append(para); cur_tok += p_tok
    if cur:
        out.append({"text": heading + "\n\n" + "\n\n".join(cur),
                    "doc": doc, "page": page, "heading": heading})
    return out

def locate_body(pg, heading):
    full = pg.extract_text() or ""
    parts = re.split(re.escape(heading), full, maxsplit=1, flags=re.I)
    return parts[1] if len(parts) == 2 else ""

def get_keywords_from_text(text: str) -> set:
    """Extract keywords from text (simple tokenization and lowercasing)."""
    return set(re.findall(r'\b\w+\b', text.lower()))

def build_chunks(pdf_path: str, head_json: str, persona: str = "", job: str = "") -> List[Dict]:
    chunks = []
    persona_job_keywords = get_keywords_from_text(persona + " " + job)

    with pdfplumber.open(pdf_path) as pdf, open(head_json, encoding="utf8") as f:
        heads = sorted(json.load(f)["headings"], key=lambda h: h["page"])
        
        # if persona_job_keywords: # Only filter if keywords are provided
        #     for h in heads:
        #         heading_text_keywords = get_keywords_from_text(h["text"])
        #         # Keep heading if there's any overlap with persona/job keywords
        #         if heading_text_keywords.intersection(persona_job_keywords):
        #             filtered_heads.append(h)
        #         else:
        #             logging.debug(f"Skipping heading (no persona/job keyword match): {h["text"]}")
        # else:
        #     filtered_heads = heads # If no keywords, keep all headings
        filtered_heads = heads # Temporarily disable filtering for testing

        for i, h in enumerate(filtered_heads):
            pg0   = h["page"] - 1
            pgN   = filtered_heads[i+1]["page"] if i+1 < len(filtered_heads) else None
            body_parts = []
            for j, pg_idx in enumerate(range(pg0, pgN or len(pdf.pages))):
                pg = pdf.pages[pg_idx]
                body_parts.append(locate_body(pg, h["text"]) if j == 0 else pg.extract_text() or "")
            paras = split_paragraphs("\n".join(body_parts))
            chunks += chunk_section(h["text"], paras,
                                    os.path.basename(pdf_path), h["page"])
    return chunks

# ───────────────────────── main routine ─────────────────────
def main(pdf_dir: str = INPUT_DIR, headings_dir: str = HEADINGS_DIR, persona: str = "", job: str = ""):
    all_chunks: List[Dict] = []
    
    # Ensure directories exist
    logging.info("Checking directories...")
    logging.info("PDF directory: %s", pdf_dir)
    logging.info("Headings directory: %s", headings_dir)
    logging.info("Output directory: %s", OUT_DIR)
    
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not os.path.exists(headings_dir):
        raise FileNotFoundError(f"Headings directory not found: {headings_dir}")
    
    for pdf in sorted(pathlib.Path(pdf_dir).glob("*.pdf")):
        head_json = pathlib.Path(headings_dir)/(pdf.stem + ".headings.json")
        if not head_json.exists():
            logging.warning("Skipping %s (no headings json)", pdf.name); continue
        logging.info("Chunking %s …", pdf.name)
        all_chunks += build_chunks(str(pdf), str(head_json), persona, job)
    logging.info("Total chunks: %d", len(all_chunks))

    # --- embed ------------------------------------------------
    vecs, bs = [], 128
    for i in tqdm(range(0, len(all_chunks), bs), desc="Embedding"):
        txt_batch = [c["text"] for c in all_chunks[i:i+bs]]
        vecs.append(model.encode(txt_batch, normalize_embeddings=True).astype("float32"))
    emb = np.vstack(vecs); dim = emb.shape[1]

    # --- FAISS ------------------------------------------------
    import faiss, pathlib as pl
    idx = faiss.IndexFlatIP(dim); idx.add(emb)
    pl.Path(OUT_DIR).mkdir(exist_ok=True)
    faiss.write_index(idx, str(pl.Path(OUT_DIR)/FAISS_FILE))
    with open(pl.Path(OUT_DIR)/CHUNKS_PKL, "wb") as f:
        pickle.dump(all_chunks, f)
    logging.info("🎉  Wrote %s (%d × %d) + %s", FAISS_FILE, len(all_chunks), dim, CHUNKS_PKL)

# ───────────────── helper for orchestrator ──────────────────
def build_index_if_needed(pdf_dir="/app/input", headings_dir="/app/output", force=False, persona: str = "", job: str = ""):
    vs = pathlib.Path("/app/vector_store")
    if force or not (vs/"index.faiss").is_file() or not (vs/"chunks.pkl").is_file():
        logging.info("Vector store missing → rebuilding …")
        main(pdf_dir, headings_dir, persona, job)
    else:
        logging.info("Vector store present – skip rebuild.")

if __name__ == "__main__":
    # This block is for standalone testing/execution of build_chunks_and_index.py
    # In the main pipeline, build_index_if_needed will be called with specific dirs.
    main()

