#!/bin/bash
# Easy deployment script for Hugging Face Spaces

echo "🚀 Deploying to Hugging Face Spaces: AQI"
echo "========================================"
echo ""

# Get Hugging Face username
read -p "Enter your Hugging Face username: " HF_USERNAME

if [ -z "$HF_USERNAME" ]; then
    echo "❌ Username is required!"
    exit 1
fi

HF_SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/AQI"

echo ""
echo "📋 Steps:"
echo "1. First, create the Space on Hugging Face:"
echo "   → Go to: https://huggingface.co/new-space"
echo "   → Space name: AQI"
echo "   → SDK: Docker"
echo "   → Click 'Create Space'"
echo ""
read -p "Have you created the Space? (y/n): " SPACE_CREATED

if [ "$SPACE_CREATED" != "y" ] && [ "$SPACE_CREATED" != "Y" ]; then
    echo "⏸️  Please create the Space first, then run this script again."
    exit 0
fi

echo ""
echo "🔗 Adding Hugging Face remote..."
# Remove existing HF remote if it exists
git remote remove hf 2>/dev/null || true
git remote add hf "$HF_SPACE_URL"

echo "✅ Remote added: $HF_SPACE_URL"
echo ""
echo "📤 Pushing to Hugging Face..."
echo "   (This may take a few minutes due to large mlruns/ directory)"
echo ""

# Push to Hugging Face
git push hf main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment started!"
    echo ""
    echo "🌐 Your Space will be available at:"
    echo "   https://${HF_USERNAME}-spaces.hf.space"
    echo ""
    echo "📊 Monitor build progress at:"
    echo "   $HF_SPACE_URL"
    echo ""
    echo "⏱️  Build typically takes 5-10 minutes."
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "   - Space not created yet"
    echo "   - Authentication required (run: huggingface-cli login)"
    echo "   - Large files (mlruns/) - may need Git LFS"
    echo ""
    echo "💡 Alternative: Upload files manually via web interface"
    echo "   Go to: $HF_SPACE_URL → Files and versions → Add file"
fi

