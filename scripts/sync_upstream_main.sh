#!/usr/bin/env bash
set -eu
set -o pipefail

REMOTE_UPSTREAM=${REMOTE_UPSTREAM:-upstream}
MAIN_BRANCH=${MAIN_BRANCH:-main}

cd "$(git rev-parse --show-toplevel)" || exit 1

if ! git remote | grep -q "^${REMOTE_UPSTREAM}$"; then
  echo "Remote '${REMOTE_UPSTREAM}' not configured. Configure it before running this script."
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty; please stash or commit before syncing."
  git status --short
  exit 1
fi

CURRENT_BRANCH=$(git symbolic-ref --short HEAD)

echo "Fetching ${REMOTE_UPSTREAM}/${MAIN_BRANCH} ..."
git fetch "${REMOTE_UPSTREAM}" "${MAIN_BRANCH}"

echo "Switching to ${MAIN_BRANCH} ..."
git checkout "${MAIN_BRANCH}"

echo "Fast-forward merging ${REMOTE_UPSTREAM}/${MAIN_BRANCH} into ${MAIN_BRANCH} ..."
git merge --ff-only "${REMOTE_UPSTREAM}/${MAIN_BRANCH}"

echo "Pushing ${MAIN_BRANCH} to origin ..."
git push origin "${MAIN_BRANCH}"

echo "Switching back to ${CURRENT_BRANCH} ..."
git checkout "${CURRENT_BRANCH}"

echo "Sync complete. You can rebase/merge oauth-subscriptions on top of ${MAIN_BRANCH}."
