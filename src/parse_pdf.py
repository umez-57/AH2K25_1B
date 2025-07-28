import os
import json
import statistics
from collections import Counter, defaultdict
import re

import pdfplumber

# ──────────────────────────────────────────────────────────
# Tunable parameters
SIZE_GAP            = 0.5    # heading if ≥ body_size + SIZE_GAP pt (reduced from 0.8)
WIDTH_RATIO_MAX     = 0.95   # skip line if it fills >95 % width (increased from 0.90)
BULLET_PREFIXES     = ("•", "-", "‣", "▪", "–")
MIN_CHARS, MAX_CHARS = 2, 150  # relaxed from 3, 120
DROP_UPPER_ONLY     = False  # changed from True to allow more headings
# ──────────────────────────────────────────────────────────


# ---------- helper utilities --------------------------------
def most_common_font_size(sizes):
    if not sizes:
        return None
    return Counter(round(s, 1) for s in sizes).most_common(1)[0][0]


def group_chars_by_line(page):
    lines = defaultdict(list)
    for ch in page.chars:
        lines[round(ch["top"], 1)].append(ch)
    return lines.values()


def line_text(chars):
    return "".join(c["text"] for c in sorted(chars, key=lambda c: c["x0"])).strip()


def is_bullet(text):
    return text.startswith(BULLET_PREFIXES)

def get_keywords_from_text(text: str) -> set:
    """Extract keywords from text (simple tokenization and lowercasing)."""
    return set(re.findall(r'\b\w+\b', text.lower()))


# ---------- core extractor ----------------------------------
def extract_headings_from_pdf(pdf_path):
    headings = []
    with pdfplumber.open(pdf_path) as pdf:
        all_sizes = [c["size"] for p in pdf.pages for c in p.chars]
        if not all_sizes:
            return headings

        body_size  = most_common_font_size(all_sizes)
        p80_size   = statistics.quantiles(all_sizes, n=100)[79]       # 80-th percentile
        p90_size   = statistics.quantiles(all_sizes, n=100)[89]       # 90-th percentile
        big_thresh = max(body_size + SIZE_GAP, p80_size)

        for pg_no, page in enumerate(pdf.pages, start=1):
            page_w = page.width

            for chars in group_chars_by_line(page):
                text = line_text(chars)
                if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                    continue

                avg_size   = statistics.mean(c["size"] for c in chars)
                width_used = (max(c["x1"] for c in chars) -
                              min(c["x0"] for c in chars)) / page_w

                # decide heading candidacy
                bold_frac = sum("Bold" in c["fontname"] or "bold" in c["fontname"]
                                for c in chars) / len(chars)

                big_font   = avg_size >= big_thresh
                very_big   = avg_size >= p90_size
                bold_hdr   = bold_frac > 0.10 and avg_size >= body_size # Reduced bold_frac threshold

                if not (big_font or bold_hdr):
                    continue

                if width_used > 0.98 and not very_big: # Increased WIDTH_RATIO_MAX
                    continue

                if DROP_UPPER_ONLY and (text.isupper() or text.isdigit()):
                    continue

                # normalize
                if is_bullet(text):
                    text = text.lstrip("•-‣▪– ").rstrip(" :–—-")
                else:
                    text = text.rstrip(" :–—-")

                headings.append({"page": pg_no, "text": text})

    return headings


# ---------- lightweight post-filter -------------------------
def postfilter(headings):
    """Drop very short non-title-case items (likely bold words in paragraphs)."""
    cleaned = []
    for h in headings:
        words = h["text"].split()
        # Relax filtering to allow more headings through
        # if len(words) <= 1:  # Only filter out single words
        #     continue
        if h["text"].islower() and len(words) <= 2:  # Only filter lowercase if very short
            continue
        cleaned.append(h)
    return cleaned


# ---------- CLI entry-point ---------------------------------
def main(persona: str = "", job: str = ""):
    root = os.getcwd()
    in_dir  = os.path.join(root, "input")
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(in_dir, fname)
        raw_heads   = extract_headings_from_pdf(pdf_path)
        clean_heads = postfilter(raw_heads)
        out_file = fname.replace(".pdf", ".headings.json")
        with open(os.path.join(out_dir, out_file), "w", encoding="utf8") as f:
            json.dump({"headings": clean_heads}, f, indent=2, ensure_ascii=False)

        print(f"✓ {fname}: {len(clean_heads)} headings kept ({len(raw_heads)-len(clean_heads)} dropped)")

    print("✅ Heading extraction finished → output/*.headings.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract headings from PDFs with persona/job-aware filtering.")
    parser.add_argument("--persona", type=str, default="", help="Persona for filtering headings.")
    parser.add_argument("--job", type=str, default="", help="Job to be done for filtering headings.")
    args = parser.parse_args()
    main(persona=args.persona, job=args.job)

