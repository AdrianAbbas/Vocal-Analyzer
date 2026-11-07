#!/usr/bin/env bash
# Install the Vocal Match analyzer (the `vocalmatch` command) for the current user.
#
# Run it from the unzipped VocalMatch-Analyzer folder:
#     bash install.sh
# (Tip: type "bash " in Terminal, then drag install.sh into the window.)
#
# What it does:
#   1. finds Python 3.12 or 3.11 (python.org, Homebrew or your PATH), the versions PyTorch
#      supports best. If there is none, it sets up a private Python 3.12 for Vocal Match
#      (downloaded with uv from GitHub, about 40 MB; nothing else on your Mac changes).
#      Other Pythons (3.13, conda's, ...) are not used unless you pass --python.
#   2. creates a private environment for Vocal Match:
#        macOS  ~/Library/Application Support/VocalMatch/venv
#        Linux  ~/.local/share/vocalmatch/venv
#      and installs the bundled vocalmatch wheel into it (numpy, scipy,
#      soundfile and jsonschema are downloaded from PyPI), then source separation:
#      Demucs + PyTorch (about 250 MB to download, about 1 GB on disk), so the analyzer can
#      take a full song and extract the vocal itself
#   3. links the `vocalmatch` command into ~/.local/bin and, if that folder is
#      not on your PATH, offers to add it to your shell profile
#   4. runs `vocalmatch selftest`: a synthetic vocal through a known EQ +
#      compressor, analysed, and checked; then a short synthetic song separated with
#      Demucs (this downloads the separation model once, about 80 MB); prints PASS or FAIL
#   5. prints the next steps
#
# Options:
#   --ci             non-interactive: no prompts, do not edit shell profiles
#   --python PATH    use this Python interpreter
#   --private-python always use the private Python 3.12 (step 1), even if 3.11/3.12 exist
#   --no-separation  skip Demucs + PyTorch: the analyzer then needs isolated vocal stems
#   --no-selftest    skip step 4
#   --no-next-steps  skip step 5 (install-mac.sh prints its own)
# (No `set -u`: macOS bash 3.2 treats an empty "$@" as unbound.)
set -eo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ci=0
python=""
selftest=1
next_steps=1
separation=1
private_python=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ci) ci=1 ;;
        --python) python="$2"; shift ;;
        --no-selftest) selftest=0 ;;
        --no-next-steps) next_steps=0 ;;
        --with-separation) separation=1 ;;  # the default; kept for older instructions
        --no-separation) separation=0 ;;
        --private-python) private_python=1 ;;
        -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
bold "1. Finding Python 3.12 or 3.11"
python_version() {  # prints X.Y.Z if $1 is Python >= 3.10 with venv
    "$1" -c 'import sys, venv; assert sys.version_info >= (3, 10); print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null
}
preferred() {  # Python 3.11 or 3.12: what PyTorch and Demucs support best
    case "$1" in 3.11.*|3.12.*) return 0 ;; *) return 1 ;; esac
}
mkdir -p "$app_dir" "$logs"
version=""
if [[ -n "$python" ]]; then
    if ! version=$(python_version "$python"); then
        fail "$python is not Python 3.10+ with the venv module"; exit 1
    fi
    preferred "$version" || warn "Python $version: PyTorch supports 3.11 and 3.12 best (source separation may not install)"
else
    if [[ $private_python == 0 ]]; then
        for candidate in python3.12 python3.11 \
                         /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
                         /usr/local/bin/python3.12 /usr/local/bin/python3.11 \
                         /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
                         /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
                         python3; do
            path=$(command -v "$candidate" 2>/dev/null || true)
            [[ -n "$path" ]] || continue
            if version=$(python_version "$path") && preferred "$version"; then python="$path"; break; fi
        done
    fi
    if [[ -z "$python" ]]; then
        # A private Python 3.12 (python-build-standalone, set up by uv) in $app_dir/python.
        case "$(uname -s)-$(uname -m)" in
            Darwin-arm64) uv_target=aarch64-apple-darwin ;;
            Darwin-x86_64) uv_target=x86_64-apple-darwin ;;
            Linux-x86_64) uv_target=x86_64-unknown-linux-gnu ;;
            Linux-aarch64) uv_target=aarch64-unknown-linux-gnu ;;
            *) uv_target="" ;;
        esac
        uv_log="$logs/install-python.txt"
        if [[ -n "$uv_target" ]]; then
            echo "  No Python 3.11 or 3.12 found: setting up a private Python 3.12 for Vocal Match (about 40 MB)..."
            uv_dir="$app_dir/uv"
            rm -rf "$uv_dir"; mkdir -p "$uv_dir"
            if curl -fsSL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$uv_target.tar.gz" \
                    -o "$uv_dir/uv.tar.gz" > "$uv_log" 2>&1 \
                && tar -xzf "$uv_dir/uv.tar.gz" -C "$uv_dir" >> "$uv_log" 2>&1 \
                && UV_PYTHON_INSTALL_DIR="$app_dir/python" "$uv_dir/uv-$uv_target/uv" python install 3.12 >> "$uv_log" 2>&1; then
                path=$(UV_PYTHON_INSTALL_DIR="$app_dir/python" "$uv_dir/uv-$uv_target/uv" python find --python-preference only-managed 3.12 2>> "$uv_log" || true)
                if [[ -n "$path" ]] && version=$(python_version "$path"); then python="$path"; fi
            fi
            rm -rf "$uv_dir"
            [[ -n "$python" ]] || warn "the private Python could not be set up (see $uv_log)"
        fi
    fi
    if [[ -z "$python" ]]; then
        # Last resort: any Python 3.10+ (for example 3.13).
        for candidate in python3.13 python3.14 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 \
                         /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
                         /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 python3; do
            path=$(command -v "$candidate" 2>/dev/null || true)
            [[ -n "$path" ]] || continue
            if version=$(python_version "$path"); then python="$path"; break; fi
        done
        [[ -z "$python" ]] || warn "using Python $version ($python): PyTorch supports 3.11 and 3.12 best"
    fi
fi
if [[ -z "$python" ]]; then
    fail "No Python 3.10 or newer was found, and a private one could not be set up (no internet?)."
    echo "       Install one, then run this script again:"
    echo "         https://www.python.org/downloads/  (the macOS installer for Python 3.12), or"
    echo "         brew install python@3.12"
    exit 1
fi
ok "$python (Python $version)"

# --------------------------------------------------------------- 2. install
bold "2. Installing into $venv"
pip_log="$logs/install-pip.txt"
: > "$pip_log"

# Runs "$@" with its output in $pip_log, printing what it is doing every 15 s.
with_progress() {
    local what="$1" pid start last=0 now line
    shift
    "$@" >> "$pip_log" 2>&1 &
    pid=$!
    start=$SECONDS
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
        now=$((SECONDS - start))
        if (( now - last >= 15 )); then
            last=$now
            line=$(grep -E '^(Collecting|Downloading|Installing|Building|  Downloading)' "$pip_log" 2>/dev/null | tail -1 | sed 's/^ *//' | cut -c1-90)
            printf '  ... %s, %d s%s\n' "$what" "$now" "${line:+ ($line)}"
        fi
    done
    wait "$pid"
}

"$python" -m venv --clear "$venv"
export PIP_PROGRESS_BAR=off PIP_DISABLE_PIP_VERSION_CHECK=1
pip=("$venv/bin/python" -m pip)
if ! with_progress "installing the analyzer" "${pip[@]}" install --upgrade pip \
        || ! with_progress "installing the analyzer" "${pip[@]}" install "$wheel"; then
    fail "pip could not install Vocal Match (needs internet access to PyPI). Log: $pip_log"
    tail -5 "$pip_log" | sed 's/^/       /'
    exit 1
fi
separation_ok=0
if [[ $separation == 1 ]]; then
    bold "2b. Installing source separation (Demucs + PyTorch)"
    echo "  Lets the analyzer take a full song and extract the vocal itself."
    echo "  About 250 MB to download and 1 GB on disk; this takes 1 to 5 minutes."
    free_kb=$(df -Pk "$app_dir" 2>/dev/null | awk 'NR==2 {print $4}')
    if [[ -n "$free_kb" ]] && (( free_kb < 3 * 1024 * 1024 )); then
        warn "only $((free_kb / 1024)) MB free on this disk; source separation needs about 1 GB (plus about 100 MB per analysed song)"
    fi
    torch_ok=1
    if [[ "$(uname -s)" == Linux ]]; then
        # The CPU-only PyTorch (PyPI's Linux wheel bundles CUDA, several GB).
        with_progress "installing PyTorch (CPU)" "${pip[@]}" install torch --index-url https://download.pytorch.org/whl/cpu \
            || with_progress "installing PyTorch" "${pip[@]}" install torch || torch_ok=0
    fi
    if [[ $torch_ok == 1 ]] && with_progress "installing Demucs + PyTorch" "${pip[@]}" install "${wheel}[separation]" \
        && "$venv/bin/python" -c 'import torch, demucs.apply, demucs.pretrained' >> "$pip_log" 2>&1; then
        separation_ok=1
        ok "source separation installed ($("$venv/bin/python" -c 'import torch, importlib.metadata as m; print("Demucs", m.version("demucs"), "/ PyTorch", torch.__version__)'))"
    else
        fail "source separation could not be installed (log: $pip_log). The analyzer works, but only with isolated vocals."
        tail -5 "$pip_log" | sed 's/^/       /'
    fi
else
    warn "source separation skipped (--no-separation): give the analyzer isolated vocal stems, not full songs"
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
if [[ $separation == 1 && $separation_ok == 0 ]]; then result=1; fi
if [[ $selftest == 1 ]]; then
    bold "4. Self-test"
    selftest_log="$logs/install-selftest.txt"
    separation_check=no
    [[ $separation_ok == 1 ]] && separation_check=yes
    set +e
    "$venv/bin/vocalmatch" selftest --separation "$separation_check" 2>&1 | tee "$selftest_log" | sed 's/^/  /'
    selftest_exit=${PIPESTATUS[0]}
    set -e
    if [[ $selftest_exit != 0 ]]; then
        fail "self-test failed (exit $selftest_exit). Log: $selftest_log"
        result=1
    fi
fi

# ----------------------------------------------------------- 5. next steps
if [[ $next_steps == 1 ]]; then
bold "5. Next steps"
cat <<'NEXT'
  Analyse a reference vocal against your own dry vocal (WAV, FLAC or MP3):
    vocalmatch analyze --reference reference_vocal.wav --dry my_vocal.wav \
                       --out preset.json --render matched.wav
  --reference can be a full song (MP3 is fine): the vocal is extracted with Demucs
  first (once per song; later runs reuse it). For an isolated vocal stem (an a cappella)
  add --separate no.
  Re-render with a (possibly hand-edited) preset:
    vocalmatch render --preset preset.json --dry my_vocal.wav --out out.wav
  Check the install at any time:   vocalmatch selftest
  Remove it:                       bash uninstall.sh
NEXT
fi
if [[ $result == 0 ]]; then
    bold "Vocal Match analyzer is installed and passed its self-test."
else
    bold "Installed, but a step failed (see above). Send the logs in $logs when reporting this."
fi
exit $result
