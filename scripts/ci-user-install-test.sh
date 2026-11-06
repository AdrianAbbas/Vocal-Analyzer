#!/usr/bin/env bash
# Reproduce what a user does with the downloaded package, then use the installed command
# the way a user would:
#   1. unzip VocalMatch-Analyzer.zip and (macOS) mark every file as a browser download
#      (com.apple.quarantine)
#   2. install with install.sh --ci (which runs `vocalmatch selftest`)
#   3. from an empty folder, with only the installed command: synth -> analyze -> validate
#      -> render, and check the outputs
#   4. uninstall.sh, and check that nothing is left
# Usage: scripts/ci-user-install-test.sh <VocalMatch-Analyzer.zip> <log dir>
set -euo pipefail
package="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
logs="$(mkdir -p "$2" && cd "$2" && pwd)"
work="$(mktemp -d)"
status=0
err() { echo "::error::$*"; status=1; }

python3 -m zipfile -e "$package" "$work"
folder="$work/VocalMatch-Analyzer"
if [[ "$(uname -s)" == Darwin ]]; then
    flag="0081;$(printf '%x' "$(date +%s)");Chrome;$(/usr/bin/uuidgen)"
    /usr/bin/find "$folder" -exec /usr/bin/xattr -w com.apple.quarantine "$flag" {} +
    /usr/bin/xattr -l "$folder/install.sh" | tee "$logs/quarantine.txt"
fi

# 2. The installer, as the user runs it.
set +e
bash "$folder/install.sh" --ci 2>&1 | tee "$logs/install.txt"
install_exit=${PIPESTATUS[0]}
set -e
[[ $install_exit == 0 ]] || err "install.sh failed (exit $install_exit)"
cmd="$HOME/.local/bin/vocalmatch"
[[ -x "$cmd" ]] || { err "install.sh did not create $cmd"; exit 1; }
grep -q "Vocal Match analyzer is installed and passed its self-test." "$logs/install.txt" || err "installer did not report success"

# 3. The installed command alone, from an empty folder (no repository on any path).
use="$(mktemp -d)"
(
    cd "$use"
    "$cmd" --version
    "$cmd" synth --out dry.wav --seed 3 --duration 6
    "$cmd" synth --out ref.wav --seed 3 --duration 6   # same take: analysis must find ~no processing
    "$cmd" analyze --reference ref.wav --dry dry.wav --out preset.json --render matched.wav --refine 10
    "$cmd" validate preset.json
    "$cmd" render --preset preset.json --dry dry.wav --out out.wav --tail 1
    python3 - <<'PY'
import json, wave
p = json.load(open("preset.json"))
assert p["schema_version"] == 1, p["schema_version"]
gains = [b["gain_db"] for b in p["effects"]["eq"]["bands"] if b["enabled"]]
assert all(abs(g) < 1.0 for g in gains), f"identity analysis produced EQ gains {gains}"
for f, seconds in (("matched.wav", 6), ("out.wav", 7)):
    w = wave.open(f)
    assert w.getnchannels() == 2 and abs(w.getnframes() / w.getframerate() - seconds) < 0.01, f
print("outputs OK:", sorted(p["effects"]), "| EQ gains", gains)
PY
) 2>&1 | tee "$logs/use.txt" || err "using the installed command failed"

# 4. Uninstall leaves nothing behind.
bash "$folder/uninstall.sh" --purge 2>&1 | tee "$logs/uninstall.txt"
[[ ! -e "$cmd" ]] || err "uninstall.sh left $cmd"
case "$(uname -s)" in
    Darwin) app_dir="$HOME/Library/Application Support/VocalMatch" ;;
    *)      app_dir="${XDG_DATA_HOME:-$HOME/.local/share}/vocalmatch" ;;
esac
[[ ! -e "$app_dir" ]] || err "uninstall.sh left $app_dir"
exit $status
