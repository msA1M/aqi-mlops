FROM python:3.9-slim

WORKDIR /app

# System dependencies for some ML/MLflow packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api ./api
COPY ui ./ui
COPY feature_store ./feature_store
COPY utils ./utils
COPY alerts ./alerts

# Copy MLflow tracking directory so the registered model can be loaded
COPY mlruns ./mlruns

# MLflow will read models from this local tracking URI
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Hugging Face sets PORT (typically 7860); Streamlit must listen on it
ENV PORT=7860

EXPOSE 8000
EXPOSE 7860

# Start FastAPI API (port 8000) and Streamlit UI (port $PORT) in the same container
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port=\$PORT --server.address=0.0.0.0"


