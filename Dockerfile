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
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --no-cache-dir llama-cpp-python==0.3.14
RUN pip install --no-cache-dir faiss_cpu==1.11.0.post1 numpy==2.3.2 pdfplumber==0.11.7 sentence_transformers==5.0.0 tqdm==4.67.1

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p input output vector_store models data

# Download models during build time (when internet is available)
RUN python download_models.py

# Download Gemma 3 GGUF model (compatible with llama-cpp-python 0.3.14)
RUN wget https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf -O /tmp/gemma-3-1b-it-Q4_K_M.gguf && \
    ls -la /tmp/gemma-3-1b-it-Q4_K_M.gguf && \
    cp /tmp/gemma-3-1b-it-Q4_K_M.gguf models/gemma-3-1b-it-Q4_K_M.gguf && \
    ls -la models/gemma-3-1b-it-Q4_K_M.gguf && \
    rm /tmp/gemma-3-1b-it-Q4_K_M.gguf

# Create a build-time processing script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== BUILD-TIME PROCESSING ==="\n\
echo "Processing PDFs and building vector store..."\n\
\n\
# Check if there are PDFs in input directory\n\
if [ -z "$(ls -A input/*.pdf 2>/dev/null)" ]; then\n\
    echo "Warning: No PDF files found in input/ directory during build"\n\
    echo "This is normal if you will mount PDFs at runtime"\n\
    exit 0\n\
fi\n\
\n\
# Parse all PDFs\n\
echo "Step 1: Parsing PDFs..."\n\
python src/parse_pdf.py\n\
\n\
# Build chunks and index\n\
echo "Step 2: Building chunks and vector index..."\n\
python src/build_chunks_and_index.py\n\
\n\
echo "Build-time processing completed!"\n\
echo "Vector store and parsed data are ready for runtime"\n\
' > /app/build_process.sh && chmod +x /app/build_process.sh

# Run build-time processing if PDFs are available
RUN /app/build_process.sh

# Create runtime entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== RUNTIME EXECUTION ==="\n\
\n\
# Check if input.json exists\n\
if [ ! -f "/app/input.json" ]; then\n\
    echo "Error: input.json not found in /app directory"\n\
    echo "Please mount input.json with your persona and job configuration"\n\
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
    echo "Please mount your PDF files to /app/input"\n\
    exit 1\n\
fi\n\
\n\
# Check if vector store exists, if not rebuild it\n\
if [ ! -f "/app/vector_store/index.faiss" ]; then\n\
    echo "Vector store not found, rebuilding..."\n\
    python src/parse_pdf.py\n\
    python src/build_chunks_and_index.py\n\
fi\n\
\n\
# Run main pipeline\n\
echo "Running main pipeline..."\n\
python main.py --input input --persona "$PERSONA" --job "$JOB" --output /app/output/output.json\n\
\n\
echo "Pipeline completed successfully!"\n\
echo "Output written to /app/output/output.json"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 
