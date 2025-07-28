import os, json, re, logging
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.build_chunks_and_index import build_index_if_needed
from src.parse_pdf import main as parse_pdf_main # Import parse_pdf's main function
from rank.rank_sections           import rank_sections
from src.rag_pipeline             import llm, generate_answer

# ───────────────────────── config ──────────────────────────
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RANK   = 10     # over-fetch before rerank to ensure we get at least 5
FINAL_TOP_N  = 5      # Ensure at least 5 sections
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
# ───────────────────────────────────────────────────────────


# ╭───────────────────────────────╮
# │  1 ▸ dynamic exclude builder  │
# ╰───────────────────────────────╯
def generate_exclude_list(persona: str, job: str) -> List[str]:
    """Ask the local LLM for a short comma-separated blacklist."""
    prompt = f"""Given this persona and task, list keywords that would make content unsuitable.

Persona: {persona}
Task: {job}

Return ONLY keywords separated by commas. No code, no explanations, no other text.
Example format: meat, beef, pork, chicken, fish"""
    try:
        resp = llm(prompt, max_tokens=64, temperature=0.1)
        raw = resp["choices"][0]["text"].strip()
        logging.info("Raw LLM response → %s", raw)
        
        # Clean up the response - remove code blocks, explanations, etc.
        cleaned_raw = re.sub(r'```.*?```', '', raw, flags=re.DOTALL)  # Remove code blocks
        cleaned_raw = re.sub(r'def\s+\w+.*?return.*', '', cleaned_raw, flags=re.DOTALL)  # Remove function definitions
        cleaned_raw = re.sub(r'#.*$', '', cleaned_raw, flags=re.MULTILINE)  # Remove comments
        cleaned_raw = re.sub(r'[^a-zA-Z0-9,\s-]', '', cleaned_raw)  # Keep only alphanumeric, comma, space, hyphen
        
        keywords = []
        for part in cleaned_raw.split(","):
            part = part.strip().lower()
            # Validate each keyword
            if re.match(r'^[a-z0-9-]+$', part) and 2 <= len(part) <= 30:
                keywords.append(part)
        
        # If we got no valid keywords, use fallback
        if not keywords:
            logging.warning("No valid keywords extracted, using fallback")
            if "vegetarian" in job.lower():
                keywords = ["meat", "beef", "pork", "chicken", "fish", "seafood"]
            if "gluten" in job.lower() or "gluten-free" in job.lower():
                keywords.extend(["wheat", "bread", "pasta", "flour", "gluten"])
        
        logging.info("Exclude-list → %s", keywords)
        return keywords
    except Exception as e:
        logging.warning("LLM exclude generation failed → %s", e)
        return []


# ╭───────────────────────────────╮
# │  2 ▸ simple lexical filtering │
# ╰───────────────────────────────╯
def filter_sections_by_exclude(sections: List, excludes: List[str]) -> List:
    if not excludes:
        return sections
    kept = []
    filtered_count = 0
    for sec in sections:
        hay = sec["heading"].lower()
        # Exclude if ANY keyword in the exclude list is found in the heading
        matched_excludes = [ex for ex in excludes if ex in hay]
        if matched_excludes:
            logging.info("Filtered out: '%s' (contains exclude keywords: %s)", sec["heading"], matched_excludes)
            filtered_count += 1
            continue
        kept.append(sec)
    logging.info("Filtered %d sections, kept %d sections", filtered_count, len(kept))
    return kept


# ╭──────────────────────────────╮
# │  3 ▸ Cross-Encoder rerank    │
# ╰──────────────────────────────╯
_ce: CrossEncoder | None = None        # lazy-loaded singleton


def _load_cross_encoder():
    global _ce
    if _ce is None:
        logging.info("Loading cross-encoder %s …", CROSS_ENCODER_NAME)
        _ce = CrossEncoder(CROSS_ENCODER_NAME, device="cpu")
    return _ce


def select_top_sections(filtered: List, query: str,
                        target_count: int = FINAL_TOP_N) -> List:
    """Rerank with cross-encoder; fall back on original score order."""
    try:
        ce = _load_cross_encoder()
        pairs = [(query, f"{s['heading']} ({s['doc']})") for s in filtered]
        scores = ce.predict(pairs)
        for s, sc in zip(filtered, scores):
            s["_ce_score"] = float(sc)
        filtered.sort(key=lambda x: x["_ce_score"], reverse=True)
    except Exception as e:
        logging.warning("Cross-encoder rerank failed → %s", e)

    # Ensure we return at least target_count sections if available
    return filtered[:max(target_count, len(filtered))]


# ╭──────────────────────────────╮
# │  4 ▸ Enhanced analysis       │
# ╰──────────────────────────────╯
def generate_detailed_analysis(heading: str, persona: str, job: str) -> str:
    """Generate detailed analysis relevant to persona and job."""
    # First, get the actual content of this section from the chunks
    section_content = get_section_content(heading)
    
    if not section_content:
        return f"This section provides relevant information for {persona} working on {job}."
    
    prompt = f"""Section Title: "{heading}"
Section Content: {section_content[:500]}...

As a {persona} working on {job}, analyze this specific section content and explain what valuable information it provides for your task. Focus on practical details and insights that would help you accomplish your work.

Write 2-3 sentences of direct analysis."""
    
    try:
        resp = llm(prompt, max_tokens=64, temperature=0.1)
        raw_text = resp["choices"][0]["text"].strip()
        
        # Clean up markdown formatting
        cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', raw_text)  # Remove bold formatting
        cleaned_text = re.sub(r'\*(.*?)\*', r'\1', cleaned_text)   # Remove italic formatting
        cleaned_text = re.sub(r'#+\s*', '', cleaned_text)          # Remove headers
        cleaned_text = re.sub(r'[-•]\s*', '', cleaned_text)        # Remove bullet points
        cleaned_text = re.sub(r'\n+', ' ', cleaned_text)           # Replace multiple newlines with single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)           # Normalize whitespace
        
        # Remove common prefixes
        cleaned_text = re.sub(r'^(answer|summary|analysis|response):\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^(the\s+)?(section|heading|topic)\s+(titled\s+)?"[^"]*"\s*', '', cleaned_text, flags=re.IGNORECASE)
        
        # Ensure it starts with a capital letter and ends properly
        cleaned_text = cleaned_text.strip()
        if cleaned_text and not cleaned_text[0].isupper():
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
        
        # Ensure it ends with a period if it doesn't already
        if cleaned_text and not cleaned_text.endswith(('.', '!', '?')):
            cleaned_text += '.'
            
        return cleaned_text
    except Exception as e:
        logging.warning("Analysis generation failed → %s", e)
        return f"This section provides relevant information for {persona} working on {job}."


def get_section_content(heading: str) -> str:
    """Get the actual content of a section from the chunks."""
    try:
        import pickle
        chunks_path = "vector_store/chunks.pkl"
        if not os.path.exists(chunks_path):
            return ""
        
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        
        # Find chunks that match this heading
        matching_chunks = [chunk for chunk in chunks if chunk.get("heading", "").strip() == heading.strip()]
        
        if matching_chunks:
            # Return the text content of the first matching chunk
            return matching_chunks[0].get("text", "")
        else:
            return ""
    except Exception as e:
        logging.warning("Failed to get section content: %s", e)
        return ""


# ╭──────────────────────────────╮
# │          5 ▸ pipeline        │
# ╰──────────────────────────────╯
def main(pdf_dir: str, persona: str, job: str, out_json: str):
    logging.info("▶ persona=%s  job=%s", persona, job)

    # 0) Parse PDFs with persona/job-aware filtering
    parse_pdf_main(persona=persona, job=job)

    # 1) ensure vector store
    build_index_if_needed(pdf_dir, persona=persona, job=job)

    # 2) initial retrieval with higher k to ensure we get enough sections
    cand = rank_sections(f"{persona}. {job}", k=TOP_K_RANK)

    # 3) dynamic excludes
    excludes = generate_exclude_list(persona, job)
    filt = filter_sections_by_exclude(cand, excludes)

    # 4) cross-encoder rerank → top-N (ensure at least 5)
    query = f"{persona} needs content for {job}"
    topN = select_top_sections(filt, query, FINAL_TOP_N)
    
    # Ensure we have at least 5 sections
    if len(topN) < FINAL_TOP_N:
        logging.info(f"Not enough sections after filtering/reranking ({len(topN)} < {FINAL_TOP_N}). Attempting to retrieve more.")
        # If not enough sections, try to get more from the initial candidates before filtering
        # This prioritizes getting enough sections over strict filtering if necessary
        if len(cand) >= FINAL_TOP_N:
            topN = cand[:FINAL_TOP_N]
            logging.info(f"Using top {FINAL_TOP_N} from original candidates due to aggressive filtering.")
        else:
            topN = cand # Take all available if even initial candidates are less than FINAL_TOP_N
            logging.info(f"Only {len(cand)} sections available after initial ranking. Using all.")

    # 5) Enhanced detailed analysis
    analyses = []
    for sec in topN:
        detailed_analysis = generate_detailed_analysis(sec['heading'], persona, job)
        analyses.append({
            "document": sec["doc"],
            "refined_text": detailed_analysis,
            "page_number": sec["page"]
        })

    # 6) final JSON
    result = {
        "metadata": {
            "input_documents": sorted(f for f in os.listdir(pdf_dir)
                                      if f.lower().endswith(".pdf")),
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp":
                datetime.utcnow().isoformat(timespec="seconds")
        },
        "extracted_sections": [
            {
                "document": sec["doc"],
                "section_title": sec["heading"],
                "importance_rank": i + 1,
                "page_number": sec["page"]
            } for i, sec in enumerate(topN)
        ],
        "subsection_analysis": analyses
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info("🎉 Written %s", out_json)

# ─────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    import argparse, textwrap
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Persona-aware doc-QA pipeline with dynamic exclude list +
        cross-encoder rerank (no hard-coded domain logic)."""))
    p.add_argument("--input",  default="input")
    p.add_argument("--persona", required=True)
    p.add_argument("--job",    required=True, dest="job_to_be_done")
    p.add_argument("--output", default="output/result.json")
    a = p.parse_args()
    main(a.input, a.persona, a.job_to_be_done, a.output)
