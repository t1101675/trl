#!/bin/bash
# Pull from GitHub via HTTPS proxy
# Usage: ./git-pull.sh [branch]

BRANCH=${1:-biatd}

https_proxy="http://deepseek:%2BogyigDac5@ss.deepseek.com:3128" \
http_proxy="http://deepseek:%2BogyigDac5@ss.deepseek.com:3128" \
git fetch origin "$BRANCH"

git checkout "$BRANCH" 2>/dev/null || true
git merge "origin/$BRANCH" --ff-only

echo "✅ Pulled branch: $BRANCH"
