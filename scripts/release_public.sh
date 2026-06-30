#!/usr/bin/env bash
# Release: squash current branch (default: develop) onto main, then push main to public remotes.
# Usage:
#   scripts/release_public.sh                # squash develop -> main, push to codeberg & github
#   scripts/release_public.sh <branch>       # squash <branch> -> main
#   scripts/release_public.sh <branch> "msg" # custom commit message
set -euo pipefail

SRC_BRANCH="${1:-develop}"
MSG="${2:-Release: squash of ${SRC_BRANCH} @ $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
PUBLIC_REMOTES=(codeberg github)

current() { git rev-parse --abbrev-ref HEAD; }
dirty()   { [ -n "$(git status --porcelain | grep -v '^??')" ]; }

if dirty; then
  echo "Working tree is dirty. Commit or stash first." >&2
  exit 1
fi

start=$(current)
echo "On branch: $start"

if [ "$start" != "$SRC_BRANCH" ]; then
  git checkout "$SRC_BRANCH"
fi

git checkout main
if git merge-base --is-ancestor "$SRC_BRANCH" main; then
  echo "$SRC_BRANCH is already contained in main. Nothing to squash." >&2
  exit 1
fi

git merge --squash "$SRC_BRANCH"
git commit -m "$MSG"

echo "Squashed $SRC_BRANCH -> main as single commit:"
git log -1 --oneline

echo
echo "Pushing main to: ${PUBLIC_REMOTES[*]}"
for r in "${PUBLIC_REMOTES[@]}"; do
  if git remote get-url "$r" >/dev/null 2>&1; then
    echo "  -> $r"
    git push "$r" main
  else
    echo "  (skip) $r not configured" >&2
  fi
done

git checkout "$start"
echo "Done. Returned to $start."
