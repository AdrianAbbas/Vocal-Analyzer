#!/usr/bin/env bash
# Stage <dist>/VocalMatch-Analyzer.zip: the vocalmatch wheel, install.sh, uninstall.sh and
# INSTALL.txt: the analyzer alone (Linux). macOS: scripts/package-mac.sh adds the plug-in.
# Usage: scripts/package.sh <dist dir>
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="${1:?dist directory}"
name="VocalMatch-Analyzer"
stage="$dist/$name"
rm -rf "$stage" "$dist/$name.zip"
mkdir -p "$stage"

# The schema ships inside the package; it must be the shared one.
cmp "$here/shared/preset.schema.json" "$here/analysis/vocalmatch/data/preset.schema.json"

python3 -m pip wheel --no-deps --wheel-dir "$stage" "$here/analysis" > "$dist/wheel-build.txt" 2>&1 || {
    cat "$dist/wheel-build.txt"; exit 1; }
install -m 755 "$here/scripts/install.sh" "$here/scripts/uninstall.sh" "$stage/"
install -m 644 "$here/packaging/INSTALL.txt" "$stage/"

# zipfile keeps the Unix permissions (install.sh stays executable).
python3 - "$dist" "$name" <<'PY'
import os, sys, zipfile
dist, name = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(os.path.join(dist, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(os.path.join(dist, name)):
        for f in sorted(files):
            path = os.path.join(root, f)
            z.write(path, os.path.relpath(path, dist))
PY
echo "Packaged $dist/$name.zip"
python3 -m zipfile -l "$dist/$name.zip"
