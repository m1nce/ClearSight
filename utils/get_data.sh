#!/bin/bash

# ===========================
# Cityscapes Dataset Downloader
# ===========================
# Last updated by Minchan Kim on 2025-02-11
#
# This scripts logs into the Cityscapes dataset website, 
# fetches the required dataset, and unzips it.
# 
# Usage: 
#   chmod +x get_data.sh
#   ./get_data.sh [packageID]
#  
# Options: 
#   packageID (optional) - The ID of the dataset package to download. If not 
#                          specified, it defaults to "1".

# Load environment variables from .env file
if [[ -f ../.env ]]; then
    export $(grep -v '^#' ../.env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# Set default package ID if not provided
PACKAGE_ID="${1:-3}"  # Default to "3" if no argument is given
COOKIE_FILE="cookies.txt"
LOGIN_URL="https://www.cityscapes-dataset.com/login/"
DOWNLOAD_URL="https://www.cityscapes-dataset.com/file-handling/?packageID=${PACKAGE_ID}"

# Step 1: Perform login and save cookies
echo "Logging in as $USERNAME..."
cd ..
wget --quiet --keep-session-cookies --save-cookies=$COOKIE_FILE --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" "$LOGIN_URL"

# Step 2: Download the dataset and save it in the data folder
OUTPUT_ZIP="dataset_${PACKAGE_ID}.zip"
echo "Downloading dataset with package ID: $PACKAGE_ID..."
wget --load-cookies=$COOKIE_FILE --content-disposition --trust-server-names -O "$OUTPUT_ZIP" "$DOWNLOAD_URL"

# Step 3: Clean up cookies and unwanted intermediate files
rm -f $COOKIE_FILE index.html
echo "Download complete! File saved to $OUTPUT_ZIP"

# Step 4: Unzip the downloaded file
echo "Unzipping the downloaded file..."
unzip -o "$OUTPUT_ZIP" -d "data/"
rm -f "$OUTPUT_ZIP"
echo "Unzipping complete! Data saved to 'data/' directory."

# Step 5: Remove unnecessary files from the dataset
echo "Cleaning up unnecessary files..."
rm -f data/index.html data/license.txt data/README*

echo "Dataset setup is complete!"