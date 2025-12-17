#!/bin/bash
# Deployment script for Hugging Face Spaces
# Space Name: AQI
# Note: Use `huggingface-cli login` to authenticate instead of hardcoding tokens

set -e

echo "🚀 Deploying to Hugging Face Spaces: AQI"
echo "=========================================="

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS not found. Installing..."
    echo "Please install Git LFS: https://git-lfs.github.com/"
    exit 1
fi

# Initialize Git LFS
echo "📦 Initializing Git LFS..."
git lfs install

# Track mlruns directory with Git LFS
echo "📦 Tracking mlruns/ with Git LFS..."
git lfs track "mlruns/**"

# Add .gitattributes
git add .gitattributes

# Add required files
echo "📝 Staging files..."
git add Dockerfile
git add README.md
git add requirements.txt
git add .gitignore
git add api/
git add ui/
git add feature_store/
git add utils/
git add alerts/

# Force add mlruns (it's in .gitignore but needed for deployment)
echo "📦 Adding mlruns/ directory (this may take a while, ~573MB)..."
git add -f mlruns/

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

echo ""
echo "✅ Ready to commit and push!"
echo ""
echo "Next steps:"
echo "1. Create a new Space on Hugging Face: https://huggingface.co/new-space"
echo "   - Name: AQI"
echo "   - SDK: Docker"
echo "2. Then run:"
echo "   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AQI"
echo "   git commit -m 'Deploy to Hugging Face Spaces'"
echo "   git push hf main"
echo ""
echo "Or use the Hugging Face CLI:"
echo "   huggingface-cli login"
echo "   huggingface-cli repo create AQI --type space --sdk docker"
echo "   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/AQI"
echo "   git push hf main"

