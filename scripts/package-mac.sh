#!/usr/bin/env bash
# Sign the built bundles ad hoc (whole bundle, --deep), verify them, and stage
# <dist>/VocalMatch-macOS-Universal.zip: the plug-in (Vocal Match.component + Vocal Match.vst3),
# the analyzer (vocalmatch wheel), install-mac.sh / install.sh, uninstall-mac.sh / uninstall.sh
# and INSTALL.txt. (As AdrianAbbas/LIMINAL's scripts/package-mac.sh for TAPEWORN FX.)
# Usage: scripts/package-mac.sh <plug-in artefacts dir, e.g. plugin/build/VocalMatch_artefacts/Release> <dist dir>
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artefacts="$1"
dist="$2"
name="VocalMatch-macOS-Universal"
stage="$dist/$name"
rm -rf "$stage" "$dist/$name.zip"
mkdir -p "$stage"

for bundle in "$artefacts/AU/Vocal Match.component" "$artefacts/VST3/Vocal Match.vst3"; do
    # Developer ID signing (if ever added) runs before this script; keep it.
    if ! /usr/bin/codesign -dv --verbose=2 "$bundle" 2>&1 | /usr/bin/grep -q "Authority=Developer ID Application"; then
        /usr/bin/codesign --force --deep --sign - --timestamp=none "$bundle"
    fi
    /usr/bin/codesign --verify --strict --verbose=4 "$bundle"
    /usr/bin/codesign -dv --verbose=2 "$bundle" 2>&1 | /usr/bin/grep -E "^(Identifier|Format|Signature|Authority|Sealed Resources)"
done
/usr/bin/ditto "$artefacts/AU/Vocal Match.component" "$stage/Vocal Match.component"
/usr/bin/ditto "$artefacts/VST3/Vocal Match.vst3" "$stage/Vocal Match.vst3"

# The analyzer, as in scripts/package.sh. The schema ships inside it; it must be the shared one.
cmp "$here/shared/preset.schema.json" "$here/analysis/vocalmatch/data/preset.schema.json"
python3 -m pip wheel --no-deps --wheel-dir "$stage" "$here/analysis" > "$dist/wheel-build-mac.txt" 2>&1 || {
    cat "$dist/wheel-build-mac.txt"; exit 1; }

install -m 755 "$here/scripts/install-mac.sh" "$here/scripts/uninstall-mac.sh" "$here/scripts/install.sh" "$here/scripts/uninstall.sh" "$stage/"
install -m 644 "$here/packaging/INSTALL-mac.txt" "$stage/INSTALL.txt"
# ditto keeps permissions and bundle metadata, unlike a plain zip of the artifact.
/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$stage" "$dist/$name.zip"
echo "Packaged $dist/$name.zip"
/usr/bin/unzip -l "$dist/$name.zip" | /usr/bin/grep -E "install|INSTALL|\.whl|MacOS/Vocal Match$"
