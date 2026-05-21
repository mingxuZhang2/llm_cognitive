#!/bin/bash
# Download HF models via wget (bypass HF hub SSL issues)
# Usage: bash download_models.sh <repo_id> <target_dir>
# Example: bash download_models.sh Qwen/Qwen2.5-0.5B-Instruct /hpc2hdd/home/mzhang630/data/nature/models/Qwen2.5-0.5B-Instruct

REPO=$1
TARGET=$2

if [ -z "$REPO" ] || [ -z "$TARGET" ]; then
    echo "Usage: $0 <repo_id> <target_dir>"
    exit 1
fi

mkdir -p "$TARGET"

BASE_URL="https://huggingface.co/${REPO}/resolve/main"

# Get file list from API
echo "Getting file list for $REPO..."
FILES=$(python3 -c "
import requests, json
r = requests.get('https://huggingface.co/api/models/${REPO}', params={'blobs': False}, timeout=30)
data = r.json()
siblings = data.get('siblings', [])
for s in siblings:
    print(s['rfilename'])
" 2>/dev/null)

if [ -z "$FILES" ]; then
    echo "ERROR: Could not get file list"
    exit 1
fi

TOTAL=$(echo "$FILES" | wc -l)
echo "Found $TOTAL files to download"

COUNT=0
for FILE in $FILES; do
    COUNT=$((COUNT + 1))
    TARGET_FILE="${TARGET}/${FILE}"
    TARGET_DIR=$(dirname "$TARGET_FILE")
    mkdir -p "$TARGET_DIR"

    if [ -f "$TARGET_FILE" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $FILE"
        continue
    fi

    echo "[$COUNT/$TOTAL] Downloading: $FILE"
    wget -q --show-progress -L "${BASE_URL}/${FILE}" -O "$TARGET_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED: $FILE"
        rm -f "$TARGET_FILE"
    fi
done

echo "Done: $REPO -> $TARGET"
