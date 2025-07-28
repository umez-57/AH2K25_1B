# PDF Processing Pipeline Docker Container

This Docker container processes PDF documents using a RAG (Retrieval-Augmented Generation) pipeline to extract relevant sections based on a persona and job description.

## Features

- Parses PDF documents to extract headings and content
- Builds vector embeddings for semantic search
- Uses cross-encoder reranking for improved relevance
- Generates detailed analysis of relevant sections
- Supports persona-aware filtering and dynamic exclude lists

## Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-processor .
```

## Usage

### 1. Prepare your input files

Create an `input.json` file with the following structure:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "create_manageable_forms",
        "description": "Creating manageable forms"
    },
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1"
        }
    ],
    "persona": {
        "role": "HR professional"
    },
    "job_to_be_done": {
        "task": "Create and manage fillable forms for onboarding and compliance."
    }
}
```

### 2. Place your PDF files

Put all your PDF files in an `input/` directory. The filenames should match those specified in the `input.json` file.

### 3. Run the container

```bash
docker run -v $(pwd)/input:/app/input -v $(pwd)/input.json:/app/input.json -v $(pwd)/output:/app/output pdf-processor
```

### 4. Get the results

The output will be written to `output.json` in your current directory.

## Input Structure

- **input.json**: Configuration file with persona, job description, and document list
- **input/**: Directory containing all PDF files referenced in input.json

## Output

The pipeline generates `output.json` with:

- **metadata**: Information about input documents, persona, job, and processing timestamp
- **extracted_sections**: Ranked list of relevant sections with document names, titles, and page numbers
- **subsection_analysis**: Detailed analysis of each relevant section

## Pipeline Steps

1. **PDF Parsing**: Extracts headings and content from all PDFs
2. **Chunk Building**: Creates semantic chunks with embeddings
3. **Index Building**: Builds FAISS vector index for similarity search
4. **Retrieval**: Finds relevant sections using semantic search
5. **Filtering**: Applies persona-aware filtering and dynamic exclude lists
6. **Reranking**: Uses cross-encoder for improved relevance ranking
7. **Analysis**: Generates detailed analysis of top sections

## Requirements


- Docker
- PDF files in input/ directory
- input.json configuration file

## Notes

- Models are downloaded during the Docker build process (when internet is available)
- The container runs completely offline once built
- All processing happens inside the container
- Output is written to the mounted output directory 