# Local Setup and Execution Guide (Without Docker)

> **When to use this guide**  
> If the Docker workflow fails or you simply prefer running everything natively, follow the steps below.  
> **Internet access is mandatory the first time** you set things up because the script downloads large open-source models.


---

## 1. Create a Python Virtual Environment

Open a terminal in the project root directory and run:

```bash
python -m venv .venv
```

- This creates a `.venv/` folder containing an isolated Python environment.

## 2. Activate the Virtual Environment

- **Windows (PowerShell)**:

  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **Windows (cmd.exe)**:

  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **macOS / Linux**:

  ```bash
  source .venv/bin/activate
  ```

## 3. Install Python Dependencies

With the venv activated, install all required libraries:

```bash
pip install -r requirements.txt
```

- This installs the following core libraries explicitly:
  - `faiss_cpu==1.11.0.post1`
  - `numpy==2.3.2`
  - `pdfplumber==0.11.7`
  - `sentence_transformers==5.0.0`
  - `tqdm==4.67.1`
  - `llama-cpp-python`

## 4. Download the LLM Model (One-Time Only)

```bash
# Create a folder to hold local models
mkdir -p models

# Download Gemma-3-1b-it-Q4_K_M (~250 MB)
wget -O models/gemma-3-1b-it-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```
After this, models/ contains gemma-3-1b-it-Q4_K_M.gguf, which llama-cpp-python loads offline.


## 5. Parse PDFs (Layout Extraction)

Run the PDF parsing engine to extract text, font, and layout data from your PDF files.

```bash
python .\src\parse_pdf.py
```

- **Input**: Reads all `*.pdf` files under the `input/` folder in the project root directory.
- **Output**: Typically writes intermediate files (e.g., extracted JSON) in a `data/` or temp folder.

## 6. Vectorize and Index Chunks

Build semantic embeddings and a FAISS index over the extracted PDF chunks:

```bash
python .\src\build_chunks_and_index.py
```

- **Embeddings**: Uses the `intfloat/e5-base-v2` model to generate 768-dimensional vectors.
- **Index**: Stores vectors and chunk metadata for fast similarity search.

## 7. Run the Main Pipeline

Execute the end-to-end persona-aware extraction and ranking:

```bash
python main.py \
  --input input \
  --persona "Food Contractor" \
  --job "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items." \
  --output output/result_food.json
```

- \`\`: Path to your PDF folder (e.g., `input/`).
- \`\`: User role you want to tailor the results for.
- \`\`: The specific task to accomplish.
- \`\`: Path and filename for the final JSON output.
- **Output files**: After running, the `output/` folder will contain intermediate heading files named `<pdf_basename>_headings.json` for each input PDF, plus the final JSON named as you specified (e.g., `result_food.json`).

You can change the `--persona` and `--job` arguments to test different scenarios on the same set of PDFs **without re-running Steps 4 and 5** — simply invoke Step 6 again with new parameters.

If you need to change the PDF dataset itself (e.g., add, remove, or replace PDFs in `input/`), then:

1. Remove all files under the `input/` directory.
2. Place your new PDF files in `input/` (ensure filenames match any required metadata).
3. Rerun **Step 4** (PDF parsing) and **Step 5** (vectorization and indexing).
4. Finally, rerun **Step 6** to generate results for the updated dataset.

This ensures the FAISS index and embeddings reflect the new PDF collection.

