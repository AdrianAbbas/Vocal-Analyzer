"""The source separation process: ``python -m vocalmatch.demucs_worker SONG OUT_DIR``.

Runs htdemucs through Demucs's Python API (no ``demucs`` command line, no torchaudio or
ffmpeg): the song is decoded with libsndfile (WAV/FLAC/MP3/OGG), resampled to the model's
rate, normalised as Demucs does, and separated in CHUNK_S chunks with CONTEXT_S of audio on
both sides (only the middle of each chunk is kept, so the seams are clean). Each chunk's
vocal stem and the rest of the song (mix - vocal) are appended to OUT_DIR/vocals.flac and
OUT_DIR/no_vocals.flac (24-bit, at separation.STEM_GAIN), so memory does not grow with the
song's length.

stdout carries ``PROGRESS <0-1> <stage>`` lines and, on a failure the user can act on, one
``ERROR <message>`` line. separation.run_demucs() starts it and reads them.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

CHUNK_S = 20.0
CONTEXT_S = 2.0


def say(line: str) -> None:
    print(line, flush=True)


def error(message: str, code: int = 3) -> int:
    say("ERROR " + message.replace("\n", " "))
    return code


def load_model(name: str):
    from demucs.pretrained import get_model

    model = get_model(name)
    model.eval()
    return model


def separate_file(song: Path, out_dir: Path, model, progress=say, device: str = "cpu") -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    from .audio_io import resample
    from .separation import STEM_GAIN

    sr = int(model.samplerate)
    channels = int(model.audio_channels)
    vi = list(model.sources).index("vocals")
    mix, fs = sf.read(str(song), dtype="float32", always_2d=True)
    if mix.shape[0] == 0:
        raise ValueError("the song has no audio")
    if mix.shape[1] < channels:
        mix = np.repeat(mix[:, :1], channels, axis=1)
    mix = mix[:, :channels]
    if fs != sr:
        mix = resample(mix, fs, sr).astype(np.float32)
    n = mix.shape[0]
    ref = mix.mean(axis=1)
    mean, std = float(ref.mean()), float(ref.std()) or 1.0
    chunk, ctx = int(CHUNK_S * sr), int(CONTEXT_S * sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_v, tmp_r = out_dir / "vocals.partial.flac", out_dir / "no_vocals.partial.flac"
    t0 = time.time()
    with sf.SoundFile(str(tmp_v), "w", sr, channels, subtype="PCM_24", format="FLAC") as fv, \
            sf.SoundFile(str(tmp_r), "w", sr, channels, subtype="PCM_24", format="FLAC") as fr:
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            lo, hi = max(0, a - ctx), min(n, b + ctx)
            seg = torch.from_numpy(((mix[lo:hi] - mean) / std).T.copy())[None]
            with torch.inference_mode():
                out = apply_model(model, seg, shifts=0, split=True, overlap=0.25, progress=False, device=device)[0]
            out = (out * std + mean).cpu().numpy()  # (sources, channels, samples)
            keep = slice(a - lo, a - lo + (b - a))
            vocals = out[vi][:, keep].T
            rest = mix[a:b] - vocals  # the two stems add up to the song
            fv.write(np.clip(vocals * STEM_GAIN, -1.0, 1.0))
            fr.write(np.clip(rest * STEM_GAIN, -1.0, 1.0))
            del out, seg
            done = b / n
            eta = (time.time() - t0) / done * (1 - done)
            progress(f"PROGRESS {0.05 + 0.95 * done:.3f} separating the vocal from the song "
                     f"({b / sr:.0f} of {n / sr:.0f} s, about {eta:.0f} s left)")
    os.replace(tmp_v, out_dir / "vocals.flac")
    os.replace(tmp_r, out_dir / "no_vocals.flac")


def main(argv: list[str] | None = None) -> int:
    try:
        faulthandler.enable(all_threads=True)  # a native crash prints where it happened
    except (AttributeError, ValueError, OSError):  # stderr without a file descriptor (tests)
        pass
    p = argparse.ArgumentParser(prog="python -m vocalmatch.demucs_worker")
    p.add_argument("song")
    p.add_argument("out_dir")
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--device", default=os.environ.get("VOCALMATCH_DEVICE", "cpu"))
    args = p.parse_args(argv)
    say("PROGRESS 0.000 loading the source separation model")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - a broken PyTorch install
        print(f"import torch: {exc!r}", file=sys.stderr)
        return error("Source separation is installed but PyTorch does not load. Run the installer again.")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "0") or 0) or min(4, os.cpu_count() or 1)))
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, device {args.device}", file=sys.stderr)
    try:
        model = load_model(args.model)
    except Exception as exc:  # noqa: BLE001 - download or load problem
        print(f"loading {args.model}: {exc!r}", file=sys.stderr)
        return error("Could not load the source separation model. It is downloaded once (about 80 MB): "
                     "check the internet connection, or run the installer again.")
    try:
        separate_file(Path(args.song), Path(args.out_dir), model, device=args.device)
    except MemoryError:
        return error("Not enough memory to separate this song. Close other apps and try again.", 4)
    except (RuntimeError, ValueError) as exc:
        if "out of memory" in str(exc).lower() or "can't allocate" in str(exc).lower():
            return error("Not enough memory to separate this song. Close other apps and try again.", 4)
        import traceback

        traceback.print_exc()
        return error(f"The source separation failed: {exc}".splitlines()[0])
    except Exception as exc:  # noqa: BLE001 - an unreadable song file
        import traceback

        traceback.print_exc()
        return error(f"Could not read the reference song ({type(exc).__name__}: {exc})".splitlines()[0])
    say("PROGRESS 1.000 separated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
