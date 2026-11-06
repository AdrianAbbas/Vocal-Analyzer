#!/usr/bin/env bash
# Download Tracktion's pluginval (a release build) for macOS into a directory; prints the
# path of its binary. Usage: scripts/ci-fetch-pluginval.sh <version, e.g. v1.0.4> <directory>
set -euo pipefail
version="${1:?version}"
dir="${2:?directory}"
mkdir -p "$dir"
curl -fsSL --retry 4 -o "$dir/pluginval.zip" "https://github.com/Tracktion/pluginval/releases/download/$version/pluginval_macOS.zip"
/usr/bin/ditto -x -k "$dir/pluginval.zip" "$dir"
binary="$dir/pluginval.app/Contents/MacOS/pluginval"
/usr/bin/xattr -dr com.apple.quarantine "$dir/pluginval.app" 2>/dev/null || true
[[ -x "$binary" ]] || { echo "pluginval binary not found in the download" >&2; exit 1; }
echo "$binary"
