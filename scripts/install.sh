#!/usr/bin/env bash
# Install the Vocal Match analyzer (the `vocalmatch` command) for the current user.
#
# Run it from the unzipped VocalMatch-Analyzer folder:
#     bash install.sh
# (Tip: type "bash " in Terminal, then drag install.sh into the window.)
#
# What it does:
#   1. finds Python 3.10 or newer (python.org, Homebrew or your PATH)
#   2. creates a private environment for Vocal Match:
#        macOS  ~/Library/Application Support/VocalMatch/venv
#        Linux  ~/.local/share/vocalmatch/venv
#      and installs the bundled vocalmatch wheel into it (numpy, scipy,
#      soundfile and jsonschema are downloaded from PyPI)
#   3. links the `vocalmatch` command into ~/.local/bin and, if that folder is
#      not on your PATH, offers to add it to your shell profile
#   4. runs `vocalmatch selftest`: a synthetic vocal through a known EQ +
#      compressor, analysed, and checked; prints PASS or FAIL
#   5. prints the next steps
#
# Options:
#   --ci             non-interactive: no prompts, do not edit shell profiles
#   --python PATH    use this Python interpreter
#   --no-selftest    skip step 4
# (No `set -u`: macOS bash 3.2 treats an empty "$@" as unbound.)
set -eo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ci=0
python=""
selftest=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ci) ci=1 ;;
        --python) python="$2"; shift ;;
        --no-selftest) selftest=0 ;;
        -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 64 ;;
    esac
    shift
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mNOTE\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

case "$(uname -s)" in
    Darwin) app_dir="$HOME/Library/Application Support/VocalMatch"; logs="$HOME/Library/Logs/VocalMatch" ;;
    Linux)  app_dir="${XDG_DATA_HOME:-$HOME/.local/share}/vocalmatch"; logs="$app_dir/logs" ;;
    *) echo "install.sh runs on macOS and Linux. (Windows: coming with the plug-in.)" >&2; exit 1 ;;
esac
venv="$app_dir/venv"
bin_dir="$HOME/.local/bin"

wheel=""
for w in "$here"/vocalmatch-*.whl; do [[ -f "$w" ]] && wheel="$w"; done
if [[ -z "$wheel" ]]; then
    echo "No vocalmatch-*.whl next to this script ($here)." >&2
    echo "Unzip VocalMatch-Analyzer.zip first and run the script from that folder." >&2
    exit 1
fi

bold "Vocal Match analyzer installer ($(basename "$wheel"))"

# ---------------------------------------------------------------- 1. Python
bold "1. Finding Python 3.10 or newer"
python_ok() {  # prints the version and succeeds if $1 is Python >= 3.10 with venv
    "$1" -c 'import sys, venv; assert sys.version_info >= (3, 10); print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null
}
if [[ -n "$python" ]]; then
    if ! version=$(python_ok "$python"); then
        fail "$python is not Python 3.10+ with the venv module"; exit 1
    fi
else
    for candidate in python3.13 python3.12 python3.11 python3.10 \
                     /opt/homebrew/bin/python3 /usr/local/bin/python3 \
                     /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
                     /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
                     /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
                     /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
                     python3; do
        path=$(command -v "$candidate" 2>/dev/null || true)
        [[ -n "$path" ]] || continue
        if version=$(python_ok "$path"); then python="$path"; break; fi
    done
fi
if [[ -z "$python" ]]; then
    fail "No Python 3.10 or newer was found (macOS's built-in python3 is 3.9, too old)."
    echo "       Install one, then run this script again:"
    echo "         https://www.python.org/downloads/  (the macOS installer), or"
    echo "         brew install python@3.12"
    exit 1
fi
ok "$python (Python $version)"

# --------------------------------------------------------------- 2. install
bold "2. Installing into $venv"
mkdir -p "$app_dir" "$logs"
pip_log="$logs/install-pip.txt"
"$python" -m venv --clear "$venv"
if ! { "$venv/bin/python" -m pip install --upgrade pip && "$venv/bin/python" -m pip install "$wheel"; } > "$pip_log" 2>&1; then
    fail "pip could not install Vocal Match (needs internet access to PyPI). Log: $pip_log"
    tail -5 "$pip_log" | sed 's/^/       /'
    exit 1
fi
installed_version=$("$venv/bin/vocalmatch" --version)
ok "$installed_version with $("$venv/bin/python" -c 'import numpy, scipy; print("numpy", numpy.__version__, "/ scipy", scipy.__version__)')"

# ----------------------------------------------------------------- 3. link
bold "3. Linking the vocalmatch command"
mkdir -p "$bin_dir"
ln -sf "$venv/bin/vocalmatch" "$bin_dir/vocalmatch"
ok "$bin_dir/vocalmatch -> $venv/bin/vocalmatch"
on_path=0
case ":$PATH:" in *":$bin_dir:"*) on_path=1 ;; esac
# shellcheck disable=SC2016  # $HOME stays literal in the profile line
path_line='export PATH="$HOME/.local/bin:$PATH"  # added by the Vocal Match installer'
if [[ $on_path == 1 ]]; then
    ok "$bin_dir is on your PATH"
else
    case "${SHELL##*/}" in
        zsh) profile="$HOME/.zshrc" ;;
        bash) [[ "$(uname -s)" == Darwin ]] && profile="$HOME/.bash_profile" || profile="$HOME/.bashrc" ;;
        *) profile="$HOME/.profile" ;;
    esac
    if [[ -f "$profile" ]] && grep -qF "added by the Vocal Match installer" "$profile"; then
        ok "$profile already adds $bin_dir to PATH (open a new Terminal window)"
    elif [[ $ci == 0 ]]; then
        printf '  %s is not on your PATH. Add it in %s? [Y/n] ' "$bin_dir" "$profile"
        read -r answer
        if [[ -z "$answer" || "$answer" == [Yy]* ]]; then
            printf '\n%s\n' "$path_line" >> "$profile"
            ok "added to $profile (open a new Terminal window to use it)"
        else
            warn "not added. Run it as $bin_dir/vocalmatch, or add this line to $profile:"
            echo "       $path_line"
        fi
    else
        warn "$bin_dir is not on PATH (--ci: profile left unchanged)"
    fi
fi

# ------------------------------------------------------------- 4. selftest
result=0
if [[ $selftest == 1 ]]; then
    bold "4. Self-test"
    selftest_log="$logs/install-selftest.txt"
    set +e
    "$venv/bin/vocalmatch" selftest > "$selftest_log" 2>&1
    selftest_exit=$?
    set -e
    sed 's/^/  /' "$selftest_log"
    if [[ $selftest_exit != 0 ]]; then
        fail "self-test failed (exit $selftest_exit). Log: $selftest_log"
        result=1
    fi
fi

# ----------------------------------------------------------- 5. next steps
bold "5. Next steps"
cat <<'NEXT'
  Analyse a reference vocal against your own dry vocal (WAV, FLAC or MP3):
    vocalmatch analyze --reference reference_vocal.wav --dry my_vocal.wav \
                       --out preset.json --render matched.wav
  Until source separation lands, --reference must be an isolated vocal stem
  (an a cappella), not a full mix.
  Re-render with a (possibly hand-edited) preset:
    vocalmatch render --preset preset.json --dry my_vocal.wav --out out.wav
  Check the install at any time:   vocalmatch selftest
  Remove it:                       bash uninstall.sh
NEXT
if [[ $result == 0 ]]; then
    bold "Vocal Match analyzer is installed and passed its self-test."
else
    bold "Installed, but the self-test failed. Send the logs in $logs when reporting this."
fi
exit $result
