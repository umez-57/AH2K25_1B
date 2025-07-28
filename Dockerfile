FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    wget \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (CPU-only versions)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --no-cache-dir llama-cpp-python==0.2.23
RUN pip install --no-cache-dir faiss_cpu==1.11.0.post1 numpy==2.3.2 pdfplumber==0.11.7 sentence_transformers==5.0.0 tqdm==4.67.1

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p input output vector_store models

# Download models during build time (when internet is available)
RUN python download_models.py

# Download Gemini GGUF model
RUN wget -O models/gemma-3-1b-it-Q4_K_M.gguf https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# Copy sample PDFs for build-time processing (we'll use the actual PDFs from the input directory)
# This ensures the pipeline components are tested during build
RUN cp input/*.pdf /tmp/ 2>/dev/null || echo "No PDFs to copy during build"

# Create a simple entrypoint that just runs main.py
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting PDF processing pipeline..."\n\
\n\
# Check if input.json exists\n\
if [ ! -f "/app/input.json" ]; then\n\
    echo "Error: input.json not found in /app directory"\n\
    exit 1\n\
fi\n\
\n\
# Parse input.json\n\
echo "Parsing input.json..."\n\
INPUT_DATA=$(cat /app/input.json)\n\
\n\
# Extract persona and job from input.json\n\
PERSONA=$(echo "$INPUT_DATA" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get(\"persona\", {}).get(\"role\", \"\"))")\n\
JOB=$(echo "$INPUT_DATA" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get(\"job_to_be_done\", {}).get(\"task\", \"\"))")\n\
\n\
echo "Persona: $PERSONA"\n\
echo "Job: $JOB"\n\
\n\
# Check if input directory has PDFs\n\
if [ ! -d "/app/input" ] || [ -z "$(ls -A /app/input/*.pdf 2>/dev/null)" ]; then\n\
    echo "Error: No PDF files found in /app/input directory"\n\
    exit 1\n\
fi\n\
\n\
# Run PDF parsing\n\
echo "Running PDF parsing..."\n\
python src/parse_pdf.py --persona "$PERSONA" --job "$JOB"\n\
\n\
# Build chunks and index\n\
echo "Building chunks and index..."\n\
python -c "from src.build_chunks_and_index import build_index_if_needed; build_index_if_needed(persona=\"$PERSONA\", job=\"$JOB\")"\n\
\n\
# Run main pipeline\n\
echo "Running main pipeline..."\n\
python main.py --input input --persona "$PERSONA" --job "$JOB" --output output.json\n\
\n\
echo "Pipeline completed successfully!"\n\
echo "Output written to output.json"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 
