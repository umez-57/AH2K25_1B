# Assumptions and Test Case Structure

This document outlines the directory layout, required files, and execution workflow for running the PDF processing solution against sample and custom test cases.

---

## 1. Input Directory Structure

```
repo_root/
├── Dockerfile
├── README.md
├── requirements.txt
├── input.json                  ← Defines PDF metadata, persona, and job-to-be-done
├── input/                      ← Contains PDF files to process
│   ├── doc1.pdf
│   └── doc2.pdf
└── output/
│   └── output.json                     
```

- ``: Located at the repository root, mandatory. Specifies `challenge_info`, list of `documents`, `persona`, and `job_to_be_done`.
- ``: Directory containing all PDF files referenced in `input.json`.

## 2. `input.json` Format

```jsonc
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner",
    "description": "France Travel"
  },
  "documents": [
    { "filename": "South of France - Cities.pdf",               "title": "South of France - Cities" },
    { "filename": "South of France - Cuisine.pdf",              "title": "South of France - Cuisine" },
    { "filename": "South of France - History.pdf",              "title": "South of France - History" },
    { "filename": "South of France - Restaurants and Hotels.pdf","title": "South of France - Restaurants and Hotels" },
    { "filename": "South of France - Things to Do.pdf",         "title": "South of France - Things to Do" },
    { "filename": "South of France - Tips and Tricks.pdf",      "title": "South of France - Tips and Tricks" },
    { "filename": "South of France - Traditions and Culture.pdf","title": "South of France - Traditions and Culture" }
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  }
}
```

- ``: Meta-information about the test case (ID, name, description).
- ``: Array of PDF entries with filenames and titles; the PDFs must exist under `input/`.
- ``: Object defining the user role for tailoring outputs.
- ``: Object defining the specific task for the persona.

## 3. Build & Vectorization

1. **Docker build** automatically vectorizes all PDFs in `input/` using our embedding models.
2. **Command** (run from `repo_root`):
   ```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" -v "$(pwd)/input.json:/app/input.json" --network none ah2k25-1b .
   ```

- No changes to directory structure are needed during build.
- The resulting image bundles PDF parsing, vector indexing, and heading extraction.

## 4. Run & Output Generation

After a successful build:

1. **Run command** (mount entire repo to expose both `input.json` and `input/`):

   ```bash
   docker run --rm \
     -v "$(pwd):/app:ro" \
     -v "$(pwd)/output:/app/output" \
     --network none \
     <your_image_tag>
   ```

2. **Behavior inside container**:

   - Reads `/app/input.json` for metadata, persona, and job.
   - Processes all PDFs under `/app/input/`.
   - Writes one JSON per PDF to `/app/output/`, named `{filename}.json`.

3. **Results**: Check `output/` on the host; it will contain your result JSON files.

## 5. Iterating Persona & Job on Same PDFs

- To test different personas or tasks **without rebuilding**, edit `input.json` at the root and rerun the **docker run** step.
- The same PDF files under `input/` will be reprocessed with updated instructions.

## 6. Custom Test Cases

To evaluate a completely new dataset:

1. **Replace** `input.json` at the repo root with your new test case’s metadata file.
2. **Replace** the contents of `input/` with your new PDF files (ensure filenames match inside `input.json`).
3. **Rebuild** the Docker image if dependencies changed (otherwise optional):
   ```bash
   docker build --platform linux/amd64 -t <your_image_tag> .
   ```
4. **Run**:
   ```bash
   docker run --rm \
     -v "$(pwd):/app:ro" \
     -v "$(pwd)/output:/app/output" \
     --network none \
     <your_image_tag>
   ```

## 7. Generic, No Hard-Coding

- The pipeline reads **only** from `/app/input.json` and `/app/input/*.pdf`; there is **no** hard-coded persona or job logic.
- This design ensures a generic, reusable solution for any PDF collection and user scenario.

