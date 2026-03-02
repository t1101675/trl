#!/bin/bash
# Push to GitHub (via HTTPS proxy)
# Usage: ./git-push.sh [commit message] [branch]

MSG=${1:-"update"}
BRANCH=${2:-biatd}

git add -A
git commit -m "$MSG" || echo "Nothing to commit"

/usr/bin/env \
    http{,s}_proxy=http://172.29.4.175:22222 \
    git push origin "$BRANCH"

echo "✅ Pushed branch: $BRANCH"
