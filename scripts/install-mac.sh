#!/usr/bin/env bash
# Install Vocal Match for the current user: the plug-in (Audio Unit + VST3) and the analyzer
# (the `vocalmatch` command), and make sure macOS will load the plug-in.
#
# Run it from the unzipped VocalMatch-macOS-Universal folder:
#     bash install-mac.sh
# (Tip: type "bash " in Terminal, then drag install-mac.sh into the window.)
#
# What it does:
#   1. copies Vocal Match.component to ~/Library/Audio/Plug-Ins/Components
#      and Vocal Match.vst3 to ~/Library/Audio/Plug-Ins/VST3
#   2. removes the com.apple.quarantine flag your browser added. Without this,
#      macOS refuses to load the plug-in and GarageBand shows a silent slot
#   3. re-signs both bundles (ad hoc) and verifies the signatures
#   4. restarts the Audio Unit registrar so hosts see the new copy
#   5. runs Apple's Audio Unit validator (auval) and prints PASS or FAIL
#   6. installs the analyzer with install.sh (Python 3.10+, the `vocalmatch` command, its self-test)
#   7. prints the next steps for GarageBand and Logic
#
# Options:
#   --ci              non-interactive: no prompts, no check for running hosts, no shell profile edits
#   --out-of-process  remove the sandboxSafe flag from the installed copy, so
#                     GarageBand hosts it in a separate process (fallback only)
#   --no-analyzer     install the plug-in only (skip step 6)
# (No `set -u`: macOS bash 3.2 treats an empty "$@" as unbound.)
set -eo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ci=0
out_of_process=0
analyzer=1
for arg in "$@"; do
    case "$arg" in
        --ci) ci=1 ;;
        --out-of-process) out_of_process=1 ;;
        --no-analyzer) analyzer=0 ;;
        -h|--help) /usr/bin/sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 64 ;;
    esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mNOTE\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

if [[ "$(uname -s)" != Darwin ]]; then
    echo "install-mac.sh runs on macOS only. (Linux: bash install.sh installs the analyzer.)" >&2
    exit 1
fi

name="Vocal Match"
component_src="$here/$name.component"
vst3_src="$here/$name.vst3"
if [[ ! -d "$component_src" ]]; then
    echo "$name.component was not found next to this script ($here)." >&2
    echo "Unzip VocalMatch-macOS-Universal.zip first and run the script from that folder." >&2
    exit 1
fi

components="$HOME/Library/Audio/Plug-Ins/Components"
vst3="$HOME/Library/Audio/Plug-Ins/VST3"
logs="$HOME/Library/Logs/VocalMatch"
mkdir -p "$components" "$vst3" "$logs"

bold "Vocal Match installer (plug-in + analyzer)"

# Hosts keep their own view of plug-ins while running.
if [[ $ci == 0 ]]; then
    running=$(/usr/bin/pgrep -l -x GarageBand 2>/dev/null; /usr/bin/pgrep -l -f 'Logic Pro' 2>/dev/null || true)
    if [[ -n "$running" ]]; then
        warn "GarageBand or Logic is running. Quit it (Cmd-Q), then press Return to continue."
        read -r _
    fi
fi

# A second, system-wide copy would compete with this one.
for system_copy in "/Library/Audio/Plug-Ins/Components/$name.component" "/Library/Audio/Plug-Ins/VST3/$name.vst3"; do
    if [[ -e "$system_copy" ]]; then
        warn "An older system-wide copy exists: $system_copy"
        warn "Remove it so hosts load the new one:  sudo rm -rf \"$system_copy\""
    fi
done

install_bundle() {
    local src="$1" dest="$2" binary
    rm -rf "$dest"
    /usr/bin/ditto "$src" "$dest"
    binary="$dest/Contents/MacOS/$name"
    chmod +x "$binary"
    /usr/bin/xattr -dr com.apple.quarantine "$dest" 2>/dev/null || true
    if /usr/bin/xattr -lr "$dest" 2>/dev/null | /usr/bin/grep -q com.apple.quarantine; then
        fail "could not remove the quarantine flag from $dest"
        return 1
    fi
    if [[ "$dest" == *.component && $out_of_process == 1 ]]; then
        /usr/libexec/PlistBuddy -c "Delete :AudioComponents:0:sandboxSafe" "$dest/Contents/Info.plist" 2>/dev/null || true
        warn "sandboxSafe removed: GarageBand will host $name out-of-process"
    fi
    # A Developer ID signed and notarised release keeps its signature; anything else
    # is re-signed ad hoc for this Mac.
    if /usr/bin/codesign -dv --verbose=2 "$dest" 2>&1 | /usr/bin/grep -q "Authority=Developer ID Application" \
        && [[ $out_of_process == 0 ]]; then
        :
    elif ! /usr/bin/codesign --force --deep --sign - "$dest" > "$logs/codesign-sign-$(basename "$dest").txt" 2>&1; then
        fail "signing failed for $dest (see $logs/codesign-sign-$(basename "$dest").txt)"
        return 1
    fi
    if ! /usr/bin/codesign --verify --strict --verbose=2 "$dest" > "$logs/codesign-$(basename "$dest").txt" 2>&1; then
        fail "signature check failed for $dest (see $logs/codesign-$(basename "$dest").txt)"
        return 1
    fi
    ok "$(basename "$dest") installed, quarantine removed, signature valid ($(/usr/bin/file -b "$binary" | /usr/bin/sed 's/Mach-O universal binary with //'))"
}

bold "1-3. Copying, removing quarantine, signing"
install_bundle "$component_src" "$components/$name.component"
if [[ -d "$vst3_src" ]]; then
    install_bundle "$vst3_src" "$vst3/$name.vst3"
else
    warn "$name.vst3 not in this folder; skipped"
fi

bold "4. Resetting the Audio Unit registry"
/usr/bin/killall -9 AudioComponentRegistrar 2>/dev/null || true
registered=0
for _ in $(seq 1 30); do
    if /usr/bin/auval -a 2>/dev/null | /usr/bin/grep -Eq '^[[:space:]]*aufx[[:space:]]+Vcmt[[:space:]]+Wnwd'; then
        registered=1
        break
    fi
    sleep 1
done
if [[ $registered == 1 ]]; then ok "macOS lists aufx Vcmt Wnwd (Windward Audio: Vocal Match)"; else fail "Vocal Match is not listed by the Audio Unit registry after 30 s"; fi

bold "5. Validating with Apple's auval"
auval_log="$logs/install-auval.txt"
set +e
/usr/bin/auval -v aufx Vcmt Wnwd > "$auval_log" 2>&1
auval_exit=$?
set -e
if [[ $auval_exit == 0 ]] && /usr/bin/grep -q "AU VALIDATION SUCCEEDED" "$auval_log"; then
    ok "auval: PASS ($(uname -m))"
    result=0
else
    fail "auval: FAIL ($(uname -m), exit $auval_exit). Full log: $auval_log"
    /usr/bin/grep -E "FATAL|ERROR|Cannot open" "$auval_log" | /usr/bin/head -5 | /usr/bin/sed 's/^/       /'
    result=1
fi
if [[ "$(uname -m)" == arm64 ]] && /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
    set +e
    /usr/bin/arch -x86_64 /usr/bin/auval -v aufx Vcmt Wnwd > "$logs/install-auval-x86_64.txt" 2>&1
    x86_exit=$?
    set -e
    if [[ $x86_exit == 0 ]]; then ok "auval: PASS (x86_64 under Rosetta)"; else warn "auval under Rosetta exited $x86_exit (only matters for Intel-mode hosts)"; fi
fi

analyzer_result=0
if [[ $analyzer == 1 ]]; then
    bold "6. Installing the analyzer (the vocalmatch command)"
    if [[ -f "$here/install.sh" ]]; then
        analyzer_args=(--no-next-steps)
        [[ $ci == 1 ]] && analyzer_args+=(--ci)
        set +e
        /bin/bash "$here/install.sh" "${analyzer_args[@]}" 2>&1 | /usr/bin/sed 's/^/  /'
        analyzer_result=${PIPESTATUS[0]}
        set -e
        if [[ $analyzer_result == 0 ]]; then ok "analyzer installed: vocalmatch"; else fail "the analyzer did not install (exit $analyzer_result); the plug-in is installed"; fi
    else
        warn "install.sh not in this folder; the analyzer was skipped"
    fi
fi

bold "7. Next steps"
cat <<'NEXT'
  Make a preset with the analyzer (Terminal):
    vocalmatch analyze --reference song.mp3 --dry my_vocal.wav --out preset.json
  (--reference: the full song if Demucs is installed (bash install.sh --with-separation),
   otherwise an isolated vocal stem. It takes one to a few minutes.)

  GarageBand
    1. Open GarageBand > Settings > Audio/MIDI and make sure "Audio Units" is ticked.
    2. Quit and reopen GarageBand.
    3. Select your vocal track (an audio track). Open Smart Controls (B), click
       Plug-ins, click an empty slot: Audio Units > Windward Audio > Vocal Match.
    4. In the Vocal Match window click "Load preset..." and choose preset.json
       (or drop preset.json on the window). The knobs move to the estimated
       settings; the orange dots mark the estimates while you edit.
    5. MATCH AMOUNT (top right) scales the whole effect, 0 % = your dry vocal.
    6. If the window shows sliders instead of the Vocal Match interface, use the view
       menu at the top of the plug-in window to switch from "Controls" to "Editor".
  Logic Pro
    1. Logic Pro > Settings > Plug-in Manager, find Windward Audio > Vocal Match. If it
       shows as failed, select it and click "Reset & Rescan Selection".
    2. On the vocal track click an Audio FX slot: Audio Units > Windward Audio > Vocal Match.
NEXT
if [[ $result == 0 && $analyzer_result == 0 ]]; then
    bold "Vocal Match is installed and validated."
else
    bold "Installed, but a check failed. Send the logs in $logs when reporting this."
    result=1
fi
exit $result
