"""Command line interface: ``python -m vocalmatch <command> ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .audio_io import load_audio, write_audio
    from .chain import render
    from .pipeline import NOT_YET_ESTIMATED, analyze

    dry, fs = load_audio(args.dry)
    reference, _ = load_audio(args.reference, target_fs=fs)
    print(
        "note: source separation (Demucs) arrives in milestone 4; --reference is analysed as-is "
        "and should be an isolated vocal stem.",
        file=sys.stderr,
    )
    preset = analyze(reference, dry, fs, reference_name=Path(args.reference).name, tempo_bpm=args.tempo)
    preset.save(args.out)
    print(f"wrote {args.out}")
    e = preset.effects
    print(f"  eq:         {sum(b.enabled for b in e.eq.bands)} bands, confidence {e.eq.confidence:.2f}")
    c = e.compressor
    print(
        f"  compressor: thr {c.threshold_db:.1f} dB, ratio {c.ratio:.2f}:1, attack {c.attack_ms:.1f} ms, "
        f"release {c.release_ms:.0f} ms, make-up {c.makeup_db:+.1f} dB, confidence {c.confidence:.2f}"
    )
    print(f"  not yet estimated (disabled): {', '.join(NOT_YET_ESTIMATED)}")
    if args.render:
        write_audio(args.render, render(preset, dry, fs, tail_s=args.tail), fs)
        print(f"wrote {args.render}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from .audio_io import load_audio, write_audio
    from .chain import render
    from .preset import Preset

    preset = Preset.load(args.preset)
    dry, fs = load_audio(args.dry)
    write_audio(args.out, render(preset, dry, fs, tail_s=args.tail), fs)
    print(f"wrote {args.out}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from .audio_io import write_audio
    from .synth import synthetic_vocal

    x = synthetic_vocal(seed=args.seed, duration_s=args.duration, fs=args.sample_rate)
    write_audio(args.out, x, args.sample_rate)
    print(f"wrote {args.out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import json

    import jsonschema

    from .preset import validate_preset_dict

    status = 0
    for path in args.presets:
        try:
            validate_preset_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            print(f"{path}: ok")
        except (jsonschema.ValidationError, ValueError) as exc:
            print(f"{path}: INVALID: {exc}", file=sys.stderr)
            status = 1
    return status


def _cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run

    print("Vocal Match self-test: synthetic vocal through a known EQ + compressor, then analysed")
    return 0 if run() else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vocalmatch", description="Estimate a vocal effect chain from a reference.")
    p.add_argument("--version", action="version", version=f"vocalmatch {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="estimate a preset from a reference vocal and your dry vocal")
    a.add_argument("--reference", required=True, help="reference vocal (WAV/FLAC/MP3); an isolated stem until M4")
    a.add_argument("--dry", required=True, help="your dry vocal")
    a.add_argument("--out", required=True, help="output preset.json")
    a.add_argument("--render", help="also render the dry vocal through the preset to this file")
    a.add_argument("--tempo", type=float, default=None, help="song tempo in BPM (optional)")
    a.add_argument("--tail", type=float, default=0.0, help="seconds of tail to render after the dry ends")
    a.set_defaults(func=_cmd_analyze)

    r = sub.add_parser("render", help="process a dry vocal with a preset")
    r.add_argument("--preset", required=True)
    r.add_argument("--dry", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tail", type=float, default=0.0)
    r.set_defaults(func=_cmd_render)

    s = sub.add_parser("synth", help="write the synthetic test vocal")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration", type=float, default=14.0)
    s.add_argument("--sample-rate", type=int, default=44100)
    s.set_defaults(func=_cmd_synth)

    v = sub.add_parser("validate", help="check preset files against shared/preset.schema.json")
    v.add_argument("presets", nargs="+")
    v.set_defaults(func=_cmd_validate)
    t = sub.add_parser("selftest", help="check the installation: analyse a synthetic vocal with known settings")
    t.set_defaults(func=_cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
