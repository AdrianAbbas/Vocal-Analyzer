#!/usr/bin/env bash
# Remove Vocal Match for the current user: the plug-in (AU + VST3) and the analyzer.
#   bash uninstall-mac.sh           remove the plug-in and the analyzer, keep the logs
#   bash uninstall-mac.sh --purge   also remove the logs (~/Library/Logs/VocalMatch)
# Your presets, songs and audio files are never touched.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
purge=0
[[ "${1:-}" == --purge ]] && purge=1

removed=0
for bundle in "$HOME/Library/Audio/Plug-Ins/Components/Vocal Match.component" "$HOME/Library/Audio/Plug-Ins/VST3/Vocal Match.vst3"; do
    if [[ -e "$bundle" ]]; then
        rm -rf "$bundle"
        echo "Removed $bundle"
        removed=1
    fi
done
[[ $removed == 1 ]] || echo "No Vocal Match plug-in found in ~/Library/Audio/Plug-Ins."

for system_copy in "/Library/Audio/Plug-Ins/Components/Vocal Match.component" "/Library/Audio/Plug-Ins/VST3/Vocal Match.vst3"; do
    if [[ -e "$system_copy" ]]; then echo "A system-wide copy remains: sudo rm -rf \"$system_copy\""; fi
done

if [[ -f "$here/uninstall.sh" ]]; then
    if [[ $purge == 1 ]]; then bash "$here/uninstall.sh" --purge; else bash "$here/uninstall.sh"; fi
fi

/usr/bin/killall -9 AudioComponentRegistrar 2>/dev/null || true
echo "Audio Unit registry reset. Restart GarageBand or Logic."
