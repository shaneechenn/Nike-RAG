# Nike RAG Assistant — Docker image for Hugging Face Spaces
FROM python:3.11-slim

WORKDIR /app

# torch needs libgomp on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (far smaller/faster than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both AI models at build time so queries are fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-mpnet-base-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the app and build the search indexes inside the image
COPY . .
RUN python load_data.py && python embed.py

# Hugging Face Spaces expects the app on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
