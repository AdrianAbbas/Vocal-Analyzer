#!/usr/bin/env bash
# Fetch one JUCE commit (no history) into a directory, for -DJUCE_PATH. CI caches it.
# Usage: scripts/ci-fetch-juce.sh directory commit
set -euo pipefail
dir="${1:?directory}"
commit="${2:?commit}"
rm -rf "$dir"
git init -q "$dir"
git -C "$dir" fetch -q --depth 1 https://github.com/juce-framework/JUCE.git "$commit"
git -C "$dir" checkout -q FETCH_HEAD
git -C "$dir" log -1 --format='JUCE %H %s'
