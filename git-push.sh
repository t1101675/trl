#!/bin/bash
# Push to GitHub (via SSH key)
# Usage: ./git-push.sh [branch] [commit message]

BRANCH=${1:-biatd}
MSG=${2:-"update"}

git add -A
git commit -m "$MSG" || echo "Nothing to commit"
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git push origin "$BRANCH"

echo "✅ Pushed branch: $BRANCH"
