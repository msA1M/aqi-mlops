# 🚀 Quick Deploy to Hugging Face Spaces

## Easiest Method: Web Interface (5 minutes)

### Step 1: Create the Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `AQI`
   - **SDK:** `Docker`
   - **Hardware:** `CPU basic` (free tier)
3. Click **Create Space**

### Step 2: Upload Files via Git (Recommended)
Once the Space is created, Hugging Face will show you commands. Use these:

```bash
# 1. Add Hugging Face remote (replace YOUR_USERNAME with your HF username)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AQI

# 2. Push to Hugging Face
git push hf main
```

**That's it!** The Space will automatically build and deploy.

---

## Alternative: Manual File Upload (If Git Push Fails)

If the git push fails due to large files, use the web interface:

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/AQI`
2. Click **Files and versions** tab
3. Click **Add file** → **Upload files**
4. Upload these files/folders:
   - `Dockerfile`
   - `README.md`
   - `requirements.txt`
   - `.gitignore`
   - `.gitattributes`
   - `api/` (entire folder)
   - `ui/` (entire folder)
   - `feature_store/` (entire folder)
   - `utils/` (entire folder)
   - `alerts/` (entire folder)
   - `mlruns/` (entire folder - this is large, ~573MB, be patient!)

---

## After Deployment

- ✅ Your Space will build automatically (takes 5-10 minutes)
- ✅ View logs in the Space's **Logs** tab
- ✅ Access your app at: `https://YOUR_USERNAME-spaces.hf.space`

---

## Troubleshooting

If build fails:
1. Check the **Logs** tab in your Space
2. Common issues:
   - Missing files (make sure `mlruns/` is uploaded)
   - Port conflicts (already handled in Dockerfile)
   - Model loading errors (check MLflow tracking URI)

