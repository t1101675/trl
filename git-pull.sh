#!/bin/bash
# Pull latest from GitHub (via SSH key)
# Usage: ./git-pull.sh [branch]

BRANCH=${1:-biatd}

GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git fetch origin "$BRANCH"
git checkout "$BRANCH"
git merge "origin/$BRANCH" --ff-only

echo "✅ Pulled branch: $BRANCH"
