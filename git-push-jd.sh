#!/bin/bash
# Push to GitHub via HTTPS proxy
# Usage: ./git-push.sh [commit message] [branch]

MSG=${1:-"update"}
BRANCH=${2:-biatd}

git add -A
git commit -m "$MSG" || echo "Nothing to commit"

https_proxy="http://deepseek:%2BogyigDac5@ss.deepseek.com:3128" \
http_proxy="http://deepseek:%2BogyigDac5@ss.deepseek.com:3128" \
git push origin "$BRANCH"

echo "✅ Pushed branch: $BRANCH"
