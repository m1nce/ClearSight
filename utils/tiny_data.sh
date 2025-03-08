#!/bin/bash

# ===========================================================
# Tiny Cityscapes Dataset Downloader (With Conda Support)
# ===========================================================
# Last updated by Minchan Kim on 2025-03-08
#
# This script fetches a Google Drive .zip file containing a 
# tiny subset of the leftImg8bit_trainvaltest.zip Cityscapes 
# dataset. 
# 
# Usage: 
#   chmod +x tiny_data.sh
#   bash tiny_data.sh

# Load environment variables from .env file
SCRIPT_DIR="$(dirname "$0")"

GDRIVE_ID="1KM98hFuqmfZ1QJInmYv0RGG_K_SNGa6y"
OUTPUT_ZIP="cityscapes.zip"
DATA_DIR="data"

# ✅ Check if gdown is available
if command -v gdown &>/dev/null; then
    USE_GDOWN=true
    echo "Using gdown for Google Drive download."
elif [[ -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/gdown" ]]; then
    USE_GDOWN=true
    GDOWN_CMD="$CONDA_PREFIX/bin/gdown"
    echo "Using Conda-installed gdown for Google Drive download."
else
    echo "gdown not found. Please install it using:"
    echo "   pip install gdown   (or use conda/mamba if in an environment)"
    exit 1
fi

# Step 1: Download the dataset
cd ..
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading dataset..."

gdown "$GDRIVE_ID" --output "$OUTPUT_ZIP" --continue

echo "Download complete! File saved to $OUTPUT_ZIP"

# Step 2: Unzip the downloaded file
echo "Unzipping the downloaded file..."
unzip -qo "$OUTPUT_ZIP"

# Detect extracted directory name
extracted_dir=$(unzip -qql "$OUTPUT_ZIP" | head -n 1 | awk '{print $4}' | cut -d '/' -f1)

# Ensure directory is valid before renaming
if [[ -d "$extracted_dir" ]]; then
    mv "$extracted_dir" "cityscapes"
    echo "Renamed $extracted_dir to cityscapes"
else
    echo "Error: Could not detect extracted folder. Please check manually."
fi

# Cleanup
rm -rf "__MACOSX"
find . -name ".DS_Store" -type f -delete
rm -f "$OUTPUT_ZIP"
echo "Removed zip file to save space."

echo "🎉 Dataset setup is complete!"
