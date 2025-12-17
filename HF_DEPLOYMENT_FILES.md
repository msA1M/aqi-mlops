# Files to Upload to Hugging Face Spaces

## Space Configuration
- **Space Name:** AQI
- **SDK:** Docker
- **Token:** `YOUR_HF_TOKEN` (use `huggingface-cli login` or set as environment variable)

## Required Files (Upload These)

### Core Files
- ✅ `Dockerfile` - Main Docker configuration
- ✅ `README.md` - With HF Spaces frontmatter (already updated)
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Git ignore rules
- ✅ `.gitattributes` - Git LFS configuration for large files

### Application Code
- ✅ `api/` - FastAPI backend (all Python files)
- ✅ `ui/` - Streamlit frontend (all Python files)
- ✅ `feature_store/` - Feature engineering code and data
- ✅ `utils/` - Utility functions
- ✅ `alerts/` - Email alert functionality

### Model Files (Large - 573MB)
- ✅ `mlruns/` - MLflow models directory (MUST be included)

## Quick Deployment Steps

### Method 1: Using Hugging Face CLI (Recommended)

```bash
# 1. Login to Hugging Face
huggingface-cli login
# Enter your Hugging Face token when prompted

# 2. Create the Space
huggingface-cli repo create AQI --type space --sdk docker

# 3. Initialize Git LFS (for large files)
git lfs install
git lfs track "mlruns/**"
git add .gitattributes

# 4. Add all files
git add Dockerfile README.md requirements.txt .gitignore
git add api/ ui/ feature_store/ utils/ alerts/
git add -f mlruns/  # Force add (it's in .gitignore)

# 5. Commit and push
git commit -m "Deploy to Hugging Face Spaces"
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AQI
git push hf main
```

### Method 2: Using Git (Manual)

```bash
# 1. Run the deployment script
./deploy_to_hf.sh

# 2. Create Space on HF website first, then:
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AQI
git commit -m "Deploy to Hugging Face Spaces"
git push hf main
```

### Method 3: Using Web Interface

1. Go to https://huggingface.co/spaces/YOUR_USERNAME/AQI
2. Upload files via drag-and-drop or file browser
3. **Important:** Upload `mlruns/` directory (it's large, be patient)

## File Structure in Repository

```
AQI/
├── Dockerfile              # Main Dockerfile
├── README.md              # With HF Spaces config
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore rules
├── .gitattributes         # Git LFS config
├── api/                   # FastAPI backend
│   ├── main.py
│   ├── aqi.py
│   └── weather.py
├── ui/                    # Streamlit frontend
│   └── app.py
├── feature_store/         # Feature engineering
│   ├── build_features.py
│   ├── build_weather_features.py
│   ├── features_v1.csv
│   └── weather/
│       └── weather_features.csv
├── utils/                 # Utilities
│   └── drift.py
├── alerts/                # Email alerts
│   └── email_alert.py
└── mlruns/                # MLflow models (573MB)
    ├── models/
    │   ├── AQI_Predictor/
    │   └── Weather_*/
    └── ...
```

## Important Notes

1. **mlruns/ is large (573MB)** - Use Git LFS or be patient when uploading
2. **Both services run in one container** - FastAPI (port 8000) + Streamlit (port 7860)
3. **No environment variables needed** - Everything is configured in Dockerfile
4. **Models load from bundled mlruns/** - Make sure it's included!

## After Deployment

- Streamlit UI will be accessible at: `https://YOUR_USERNAME-spaces.hf.space`
- FastAPI docs at: `https://YOUR_USERNAME-spaces.hf.space:8000/docs` (internal)
- Build logs available in HF Spaces interface

