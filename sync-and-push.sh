#!/bin/bash
# Sync from your working copy (8502) into the GitHub repo, then push.
# Usage:
#   ./sync-and-push.sh "commit message"
#
# Working copy:  blockait-main/Blockait Project  (localhost:8502)
# GitHub repo:   /Users/tair/Documents/blockait

set -e
SRC="/Users/tair/Documents/blockait-main/Blockait Project"
DST="/Users/tair/Documents/blockait"
MSG="${1:-Update Blockait}"

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC"
  exit 1
fi

echo "Syncing from $SRC → $DST"
cp "$SRC/app.py" "$SRC/pipeline.py" "$SRC/features.py" "$SRC/keywords.py" \
   "$SRC/train.py" "$SRC/requirements.txt" "$SRC/requirements-sbert.txt" "$DST/"
cp -r "$SRC/models/"* "$DST/models/"
cp -r "$SRC/data/"* "$DST/data/"

cd "$DST"
git add -A
if git diff --cached --quiet; then
  echo "Nothing changed — already up to date."
  exit 0
fi
git commit -m "$MSG"
git push origin main
echo "Done — https://github.com/oyanquantum/blockait"
