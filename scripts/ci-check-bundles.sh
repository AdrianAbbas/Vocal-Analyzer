#!/usr/bin/env bash
# CI gate for the built macOS bundles: both architecture slices, the AU identity and
# sandboxSafe in Info.plist, the bundle identifier, and the minimum macOS version.
# Usage: scripts/ci-check-bundles.sh <plug-in artefacts, e.g. plugin/build/VocalMatch_artefacts/Release>
set -euo pipefail
artefacts="$1"
status=0
for bundle in "$artefacts/AU/Vocal Match.component" "$artefacts/VST3/Vocal Match.vst3"; do
    binary="$bundle/Contents/MacOS/Vocal Match"
    if [[ ! -f "$binary" ]]; then
        echo "::error::missing $binary"; status=1; continue
    fi
    archs=$(/usr/bin/lipo -archs "$binary")
    printf '%-70s %s\n' "$binary" "$archs"
    if [[ " $archs " != *" arm64 "* || " $archs " != *" x86_64 "* ]]; then
        echo "::error::$binary must contain arm64 and x86_64 slices, has: $archs"; status=1
    fi
    for arch in x86_64 arm64; do
        minos=$(/usr/bin/otool -arch "$arch" -l "$binary" | /usr/bin/awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
        printf '    %-7s minimum macOS %s\n' "$arch" "${minos:-unknown}"
    done
    id=$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$bundle/Contents/Info.plist" 2>/dev/null || echo "(absent)")
    printf 'CFBundleIdentifier %-30s %s\n' "$(basename "$bundle")" "$id"
    if [[ "$id" != "com.windwardaudio.vocalmatch" ]]; then
        echo "::error::$(basename "$bundle") CFBundleIdentifier expected com.windwardaudio.vocalmatch, got $id"; status=1
    fi
done

plist="$artefacts/AU/Vocal Match.component/Contents/Info.plist"
check() {
    local key="$1" expected="$2" actual
    actual=$(/usr/libexec/PlistBuddy -c "Print :AudioComponents:0:$key" "$plist" 2>/dev/null || echo "(absent)")
    if [[ "$actual" == "$expected" ]]; then
        printf 'Info.plist %-16s %s\n' "$key" "$actual"
    else
        echo "::error::Info.plist $key expected '$expected', got '$actual'"; status=1
    fi
}
check type aufx
check subtype Vcmt
check manufacturer Wnwd
check factoryFunction VocalMatchAUFactory
check sandboxSafe true
check name "Windward Audio: Vocal Match"
exit $status
