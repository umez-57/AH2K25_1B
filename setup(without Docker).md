# Local Setup and Execution Guide (Without Docker)

Follow these steps to install dependencies and run the PDF processing pipeline locally (no Docker required).

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

## 4. Parse PDFs (Layout Extraction)

Run the PDF parsing engine to extract text, font, and layout data from your PDF files.

```bash
python .\src\parse_pdf.py
```

- **Input**: Reads all `*.pdf` files under the `input/` folder in the project root directory.
- **Output**: Typically writes intermediate files (e.g., extracted JSON) in a `data/` or temp folder.

## 5. Vectorize and Index Chunks

Build semantic embeddings and a FAISS index over the extracted PDF chunks:

```bash
python .\src\build_chunks_and_index.py
```

- **Embeddings**: Uses the `intfloat/e5-base-v2` model to generate 768-dimensional vectors.
- **Index**: Stores vectors and chunk metadata for fast similarity search.

## 6. Run the Main Pipeline

Execute the end-to-end persona-aware extraction and ranking:

```bash
python main.py \
  --input input \
  --persona "Food Contractor" \
  --job "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items." \
  --output output/result_food.json
```

- ``: Path to your PDF folder (e.g., `input/`).
- ``: User role you want to tailor the results for.
- ``: The specific task to accomplish.
- ``: Path and filename for the final JSON output.
- **Output files**: After running, the `output/` folder will contain intermediate heading files named `<pdf_basename>_headings.json` for each input PDF, plus the final JSON named as you specified (e.g., `result_food.json`).

You can change the `--persona` and `--job` arguments to test different scenarios on the same set of PDFs.\
If you need to process a new dataset (different PDF collection), update the `input/` folder and rerun steps 4–6.

---

### Notes and Troubleshooting

- Ensure your virtual environment is active before running any Python commands.
- If you encounter import errors, verify that `requirements.txt` lists all dependencies and reinstall them.
- The `main.py` script will produce a structured JSON matching the schema.

