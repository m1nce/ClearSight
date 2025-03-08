#!/bin/bash

# ===========================================================
# Cityscapes Dataset Downloader (With Conda Support)
# ===========================================================
# Last updated by Minchan Kim on 2025-03-08
#
# This script logs into the Cityscapes dataset website, 
# fetches the required dataset, and unzips it. Renames the 
# directory to 'cityscapes' and cleans up unnecessary files.
# 
# Usage: 
#   chmod +x get_data.sh
#   bash get_data.sh [packageID]
#  
# Options: 
#   packageID (optional) - The ID of the dataset package to download. 
#                          Defaults to "3" if not specified.

# Load environment variables from .env file
SCRIPT_DIR="$(dirname "$0")"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ -f "$ENV_FILE" ]]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# Set default package ID if not provided
PACKAGE_ID="${1:-3}"  # Default to "3" if no argument is given
COOKIE_FILE="cookies.txt"
LOGIN_URL="https://www.cityscapes-dataset.com/login/"
DOWNLOAD_URL="https://www.cityscapes-dataset.com/file-handling/?packageID=${PACKAGE_ID}"
OUTPUT_ZIP="dataset_${PACKAGE_ID}.zip"
DATA_DIR="data"

# ✅ Detect if aria2c is available (Check both global and Conda)
if command -v aria2c &>/dev/null; then
    USE_ARIA2=true
    ARIA2C_CMD="aria2c"
    echo "Using system-installed aria2 for faster downloads!"
elif [[ -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/aria2c" ]]; then
    USE_ARIA2=true
    ARIA2C_CMD="$CONDA_PREFIX/bin/aria2c"
    echo "Using Conda-installed aria2 for faster downloads!"
else
    USE_ARIA2=false
    echo "aria2 not found. Falling back to wget."
fi

# Step 1: Perform login and save cookies
echo "Logging in as $USERNAME..."
cd ..
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
wget --quiet --keep-session-cookies --save-cookies="$COOKIE_FILE" --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" "$LOGIN_URL"

# Step 2: Download the dataset
echo "Downloading dataset with package ID: $PACKAGE_ID..."

if $USE_ARIA2; then
    "$ARIA2C_CMD" --max-connection-per-server=16 --split=16 --continue=true \
        --out="$OUTPUT_ZIP" --load-cookies="$COOKIE_FILE" "$DOWNLOAD_URL"
else
    wget --load-cookies="$COOKIE_FILE" --content-disposition -O "$OUTPUT_ZIP" "$DOWNLOAD_URL" \
        --retry-connrefused --waitretry=5 --timeout=60 --tries=50 --continue
fi

# Step 3: Clean up cookies and unwanted intermediate files
rm -f "$COOKIE_FILE" index.html
echo "Download complete! File saved to $OUTPUT_ZIP"

# Step 4: Unzip the downloaded file
echo "Unzipping the downloaded file..."
unzip -qo "$OUTPUT_ZIP" | tee /dev/null
rm -f "$OUTPUT_ZIP"
echo "Unzipping complete! Data saved to '$DATA_DIR/' directory."

# Step 5: Rename downloaded file to cityscapes
extracted_dir=$(ls -td */ | head -n 1 | tr -d '/')
mv "$extracted_dir" "cityscapes"
echo "Renamed $extracted_dir directory to cityscapes directory"

# Step 6: Remove unnecessary files from the dataset
echo "Cleaning up unnecessary files..."
rm -f "index.html" "license.txt" README*

echo "🎉 Dataset setup is complete!"
