# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System packages needed to compile FAISS, scikit-learn wheels and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure output directories exist (they may be empty on first run)
RUN mkdir -p data knowledge_base picture

# Streamlit configuration via environment variables
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light

EXPOSE 8501

# Lightweight health check — polls the Streamlit internal endpoint
# start-period=120s accounts for HuggingFace model loading on cold start
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Files required in the image:
#   app.py            — Streamlit web application
#   embeddings.py     — SentenceTransformerWrapper (shared by step3 + app)
#   step1_youtube.py  — YouTube data collection & sentiment analysis
#   step2_gps.py      — GNSS trajectory analysis & GPS hotspot generation
#   step3_rag.py      — Knowledge base construction & FAISS index building
#   config_pois.json  — POI catalogue (24 Chiang Mai points of interest)
#   requirements.txt  — Python dependency list
CMD ["streamlit", "run", "app.py", \
     "--server.headless=true", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]