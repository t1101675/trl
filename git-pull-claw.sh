#!/bin/bash
# Pull from GitHub via HTTPS proxy
# Usage: ./git-pull.sh [branch]

BRANCH=${1:-biatd}

https_proxy="http://172.29.4.175:22222" \
http_proxy="http://172.29.4.175:22222" \
git fetch origin "$BRANCH"

git checkout "$BRANCH" 2>/dev/null || true
git merge "origin/$BRANCH" --ff-only

echo "✅ Pulled branch: $BRANCH"
