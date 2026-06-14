#!/bin/bash
# Push Blockait to GitHub — run from this folder in Cursor terminal:
#   cd "/Users/tair/Documents/blockait"
#   ./push.sh "your commit message"
#
# Or manually:
#   git add -A && git commit -m "update" && git push origin main

set -e
cd "$(dirname "$0")"
MSG="${1:-Update Blockait}"

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi
git commit -m "$MSG"
git push origin main
echo "Done — pushed to https://github.com/oyanquantum/blockait"
