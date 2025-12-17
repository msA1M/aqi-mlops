# Hugging Face Spaces Deployment Guide

## Space Name: AQI

## Files to Upload to Hugging Face Spaces

### Required Files (Must Upload):

1. **Dockerfile** - Main Dockerfile for the application
2. **README.md** - With HF Spaces frontmatter configuration
3. **requirements.txt** - Python dependencies
4. **api/** - FastAPI application code
5. **ui/** - Streamlit UI code
6. **feature_store/** - Feature store code and data
7. **utils/** - Utility functions
8. **alerts/** - Email alert functionality
9. **mlruns/** - MLflow models (573MB - use Git LFS or force add)

### Optional Files (Recommended):

- **test_*.py** - Test scripts
- **test_*.json** - Test data

## Deployment Steps

### Option 1: Using Git (Recommended)

1. **Initialize Git LFS for large files:**
   ```bash
   git lfs install
   git lfs track "mlruns/**"
   ```

2. **Add and commit files:**
   ```bash
   git add .gitattributes
   git add Dockerfile README.md requirements.txt
   git add api/ ui/ feature_store/ utils/ alerts/
   git add -f mlruns/  # Force add (it's in .gitignore)
   git commit -m "Deploy to Hugging Face Spaces"
   ```

3. **Push to Hugging Face:**
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AQI
   git push hf main
   ```

### Option 2: Using Hugging Face Web Interface

1. Go to https://huggingface.co/spaces/YOUR_USERNAME/AQI
2. Upload files via the web interface
3. Make sure to upload `mlruns/` directory (it's large, be patient)

## Important Notes

- The Dockerfile runs both FastAPI (port 8000) and Streamlit (port 7860) in the same container
- Streamlit will be accessible on the default HF Spaces port (7860)
- FastAPI API will be available internally at localhost:8000
- The UI connects to the API at `http://127.0.0.1:8000` (same container)

## Environment Variables

No additional environment variables needed - everything is configured in the Dockerfile.

## Model Loading

The MLflow models are loaded from the bundled `mlruns/` directory. Make sure this directory is included in your deployment.

