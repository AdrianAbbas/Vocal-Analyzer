#!/usr/bin/env bash
# Remove the Vocal Match analyzer for the current user.
#   bash uninstall.sh          remove the program, the command link and the PATH line it added
#   bash uninstall.sh --purge  also remove its logs
# Your presets, stems and audio files are never touched.
set -eo pipefail
purge=0
[[ "${1:-}" == --purge ]] && purge=1

case "$(uname -s)" in
    Darwin) app_dir="$HOME/Library/Application Support/VocalMatch"; logs="$HOME/Library/Logs/VocalMatch" ;;
    *)      app_dir="${XDG_DATA_HOME:-$HOME/.local/share}/vocalmatch"; logs="$app_dir/logs" ;;
esac

link="$HOME/.local/bin/vocalmatch"
if [[ -L "$link" ]]; then rm -f "$link"; echo "Removed $link"; fi
if [[ -d "$app_dir/venv" ]]; then rm -rf "$app_dir/venv"; echo "Removed $app_dir/venv"; fi
if [[ $purge == 1 ]]; then
    rm -rf "$logs" "$app_dir"
    echo "Removed $logs"
fi
rmdir "$app_dir" 2>/dev/null || true

for profile in "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile"; do
    if [[ -f "$profile" ]] && grep -qF "added by the Vocal Match installer" "$profile"; then
        grep -vF "added by the Vocal Match installer" "$profile" > "$profile.vocalmatch-tmp"
        mv "$profile.vocalmatch-tmp" "$profile"
        echo "Removed the PATH line from $profile"
    fi
done
echo "Vocal Match analyzer removed."
