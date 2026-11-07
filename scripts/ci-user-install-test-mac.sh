#!/usr/bin/env bash
# Reproduce what a Mac user does with the downloaded package, then validate like a host.
# (As AdrianAbbas/LIMINAL's scripts/ci-user-install-test.sh for TAPEWORN FX.)
#   1. unzip VocalMatch-macOS-Universal.zip and mark every file as a browser download
#      (com.apple.quarantine)
#   2. record whether macOS on this runner refuses the quarantined copy (informational)
#   3. install with install-mac.sh --ci: plug-in + analyzer
#   4. the AU registry lists aufx Vcmt Wnwd once, as "Windward Audio: Vocal Match";
#      auval (strict where supported) on arm64 and on x86_64 under Rosetta
#   4b. pluginval (strictness 5) on the installed AU and VST3, when its binary is given
#   5. the installed analyzer, from an empty folder: synth -> analyze -> validate; the
#      preset it wrote is loaded by the plug-in's own processor (VocalMatchRender, optional),
#      and the plug-in's Analyze code finds and runs the installed analyzer by itself
#   6. uninstall-mac.sh --purge leaves nothing behind
# Usage: scripts/ci-user-install-test-mac.sh <VocalMatch-macOS-Universal.zip> <log dir> [VocalMatchRender] [pluginval]
set -euo pipefail
package="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
logs="$(mkdir -p "$2" && cd "$2" && pwd)"
render="${3:-}"
[[ -z "$render" ]] || render="$(cd "$(dirname "$render")" && pwd)/$(basename "$render")"
pluginval="${4:-}"
work="$(mktemp -d)"
status=0
err() { echo "::error::$*"; status=1; }

# Run a command with a time limit (a Gatekeeper prompt would otherwise block forever).
limit() { local seconds="$1"; shift; /usr/bin/perl -e 'alarm shift @ARGV; exec @ARGV or die "exec failed: $!\n"' "$seconds" "$@"; }

# 1. The download.
/usr/bin/ditto -x -k "$package" "$work"
folder="$work/VocalMatch-macOS-Universal"
flag="0081;$(printf '%x' "$(date +%s)");Chrome;$(/usr/bin/uuidgen)"
/usr/bin/find "$folder" -exec /usr/bin/xattr -w com.apple.quarantine "$flag" {} +
{
    echo "Gatekeeper: $(/usr/sbin/spctl --status 2>&1)"
    /usr/bin/xattr -l "$folder/Vocal Match.component/Contents/MacOS/Vocal Match"
} | tee "$logs/quarantine.txt"

# 2. Quarantined copy installed by hand, without the script (informational).
components="$HOME/Library/Audio/Plug-Ins/Components"
vst3_dir="$HOME/Library/Audio/Plug-Ins/VST3"
mkdir -p "$components" "$vst3_dir"
rm -rf "$components/Vocal Match.component"
/usr/bin/ditto "$folder/Vocal Match.component" "$components/Vocal Match.component"
/usr/bin/killall -9 AudioComponentRegistrar 2>/dev/null || true
sleep 3
set +e
limit 120 /usr/bin/auval -v aufx Vcmt Wnwd > "$logs/auval-quarantined-without-script.txt" 2>&1
quarantined_exit=$?
set -e
echo "auval on the quarantined copy (no install script): exit $quarantined_exit" | tee -a "$logs/quarantine.txt"
rm -rf "$components/Vocal Match.component"

# 3. The user's path: the installer from the quarantined download.
set +e
limit 900 /bin/bash "$folder/install-mac.sh" --ci 2>&1 | tee "$logs/install-mac.txt"
install_exit=${PIPESTATUS[0]}
set -e
echo "install-mac.sh exit $install_exit"
[[ $install_exit == 0 ]] || err "install-mac.sh failed (exit $install_exit)"
grep -q "Vocal Match is installed and validated." "$logs/install-mac.txt" || err "install-mac.sh did not report success"
for installed in "$components/Vocal Match.component" "$vst3_dir/Vocal Match.vst3"; do
    if [[ ! -d "$installed" ]]; then err "install-mac.sh did not install $installed"; continue; fi
    if /usr/bin/xattr -lr "$installed" | /usr/bin/grep -q com.apple.quarantine; then err "installed $(basename "$installed") is still quarantined"; fi
    /usr/bin/codesign --verify --strict "$installed" || err "installed $(basename "$installed") has an invalid signature"
done

# 4. Registry and auval on the installed copy.
/usr/bin/auval -a 2>&1 | /usr/bin/grep -E 'Vcmt' | tee "$logs/registry.txt" || true
entries=$(/usr/bin/grep -c -E '^[[:space:]]*aufx[[:space:]]+Vcmt[[:space:]]+Wnwd' "$logs/registry.txt" || true)
[[ "$entries" == 1 ]] || err "the AU registry lists aufx Vcmt Wnwd $entries times (expected once)"
/usr/bin/grep -q 'Windward Audio: Vocal Match' "$logs/registry.txt" || err "the AU registry does not name it 'Windward Audio: Vocal Match'"
run_check() {
    local name="$1"; shift
    set +e
    limit 600 "$@" > "$logs/$name.txt" 2>&1
    local code=$?
    set -e
    echo "$name: exit $code"
    if [[ $code != 0 ]] || ! /usr/bin/grep -q "AU VALIDATION SUCCEEDED" "$logs/$name.txt"; then
        err "$name failed (exit $code)"
        tail -30 "$logs/$name.txt"
    fi
}
strict=""  # becomes -strict when this auval supports it
/usr/bin/auval -h 2>&1 | /usr/bin/grep -qE -- '(^|[[:space:]])-strict([[:space:]]|$)' && strict="-strict"
# shellcheck disable=SC2086  # $strict is empty or one flag
run_check auval-arm64 /usr/bin/arch -arm64 /usr/bin/auval $strict -v aufx Vcmt Wnwd
if /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
    # shellcheck disable=SC2086
    run_check auval-x86_64 /usr/bin/arch -x86_64 /usr/bin/auval $strict -v aufx Vcmt Wnwd
else
    err "Rosetta unavailable: the x86_64 slice was not validated"
fi

# 4b. pluginval on the installed copies (in-process: one process, one log per format).
if [[ -n "$pluginval" ]]; then
    for target in "$components/Vocal Match.component" "$vst3_dir/Vocal Match.vst3"; do
        log="$logs/pluginval-$(basename "$target" | tr ' ' '-').txt"
        set +e
        limit 900 "$pluginval" --strictness-level 5 --validate-in-process --timeout-ms 120000 --validate "$target" > "$log" 2>&1
        code=$?
        set -e
        echo "pluginval $(basename "$target"): exit $code"
        if [[ $code != 0 ]] || ! /usr/bin/grep -q "SUCCESS" "$log"; then
            err "pluginval failed on $(basename "$target") (exit $code)"
            tail -40 "$log"
        fi
    done
fi

# 5. The installed analyzer alone, from an empty folder, and its preset in the plug-in.
cmd="$HOME/.local/bin/vocalmatch"
if [[ -x "$cmd" ]]; then
    use="$(mktemp -d)"
    (
        cd "$use"
        "$cmd" --version
        "$cmd" synth --out dry.wav --seed 3 --duration 6
        "$cmd" synth --out ref.wav --seed 4 --duration 6
        "$cmd" analyze --reference ref.wav --dry dry.wav --out preset.json --render matched.wav --refine 10
        "$cmd" validate preset.json
        if [[ -n "$render" ]]; then
            "$render" --preset preset.json --in dry.wav --out plugin.wav
            "$cmd" render --preset preset.json --dry dry.wav --out python.wav
            "$HOME/Library/Application Support/VocalMatch/venv/bin/python" - <<'PY'
import numpy as np, soundfile as sf
a, _ = sf.read("plugin.wav", always_2d=True)
b, _ = sf.read("python.wav", always_2d=True)
r = 20 * np.log10(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
print(f"analyzer preset: plug-in vs analyzer render residual {r:.1f} dB")
assert r < -40, r
PY
            # The plug-in's Analyze code finds the installed analyzer by itself (no hints) and
            # runs it as a separate process, then loads its preset.
            env -u VOCALMATCH_ANALYZER "$render" --analyze ref.wav dry.wav --analyze-out plugin-analyze.json \
                --analyze-refine 5 --analyze-passes 1 | tee plugin-analyze.txt
            grep -q "Library/Application Support/VocalMatch/venv/bin/vocalmatch\|.local/bin/vocalmatch" plugin-analyze.txt
            "$cmd" validate plugin-analyze.json
        fi
    ) 2>&1 | tee "$logs/use.txt" || err "using the installed analyzer failed"
else
    err "install-mac.sh did not install the analyzer ($cmd)"
fi

# 6. Uninstall leaves nothing behind.
bash "$folder/uninstall-mac.sh" --purge 2>&1 | tee "$logs/uninstall.txt"
for left in "$components/Vocal Match.component" "$vst3_dir/Vocal Match.vst3" "$cmd" "$HOME/Library/Application Support/VocalMatch"; do
    [[ ! -e "$left" ]] || err "uninstall-mac.sh left $left"
done
exit $status
