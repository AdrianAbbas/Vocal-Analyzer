"""Command line interface: ``python -m vocalmatch <command> ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _load_reference_stem(args, fs: int, progress, span: float = 0.5) -> tuple:
    """(stem (n, ch), accompaniment or None, separation diagnostics, used_demucs) in the song's
    timeline at *fs*. The separation reports progress over [0, *span*]. Raises
    SeparationUnavailable (no Demucs and the reference looks like a full mix) or
    SeparationFailed."""
    from .audio_io import load_audio
    from .separation import NOT_INSTALLED, SeparationUnavailable, demucs_available, load_stem, looks_like_full_mix, separate

    use_demucs = args.separate == "yes" or (args.separate == "auto" and demucs_available())
    if not use_demucs:
        progress(0.0, "reading the reference")
        reference, _ = load_audio(args.reference, target_fs=fs)
        if args.separate == "auto":
            mix, feats = looks_like_full_mix(reference, fs)
            if mix:
                print(f"note: the reference looks like a full mix ({100 * feats['low_80hz']:.0f} % of its energy below "
                      f"80 Hz, {100 * feats['silent']:.0f} % silence) and Demucs is not installed. If it is an isolated "
                      "vocal, pass --separate no.", file=sys.stderr)
                raise SeparationUnavailable(NOT_INSTALLED)
            print("note: Demucs is not installed, so --reference is analysed as-is (it looks like an isolated vocal).",
                  file=sys.stderr)
        return reference, None, {}, False
    progress(0.0, "separating the vocal from the song")
    vocals_path, rest_path, cached = separate(args.reference, progress=lambda f, st: progress(f * span, st))
    print(f"vocal stem: {vocals_path}{' (cached: separated before)' if cached else ''}", flush=True)
    progress(0.05 if cached else span, "reading the separated vocal")
    vocals, _ = load_stem(vocals_path, target_fs=fs)
    rest, _ = load_stem(rest_path, target_fs=fs)
    return vocals, rest, {"stem": str(vocals_path), "cached": cached}, True


def _peak_memory() -> str:
    """'peak memory: analyzer N MB, separation M MB' (for the logs)."""
    try:
        import resource
    except ImportError:
        return "peak memory: unknown"
    scale = 1 / (1 << 20) if sys.platform == "darwin" else 1 / 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return f"peak memory: analyzer {own:.0f} MB" + (f", source separation {kids:.0f} MB" if kids > 1 else "")


def _singer_ids(text: str | None, singer_map) -> list[str]:
    if not text:
        return []
    if text.strip().lower() == "all":
        return [s.id for s in singer_map.singers]
    ids = [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
    for i in ids:
        singer_map.index(i)  # raises ValueError for an unknown singer
    return ids


def _print_preset(preset) -> None:
    e = preset.effects
    print(f"  eq:         {sum(b.enabled for b in e.eq.bands)} bands, confidence {e.eq.confidence:.2f}")
    c = e.compressor
    print(
        f"  compressor: thr {c.threshold_db:.1f} dB, ratio {c.ratio:.2f}:1, attack {c.attack_ms:.1f} ms, "
        f"release {c.release_ms:.0f} ms, make-up {c.makeup_db:+.1f} dB, confidence {c.confidence:.2f}"
    )
    s, d, dl, r, w = e.saturation, e.deesser, e.delay, e.reverb, e.doubler

    def state(effect, text):
        return f"{text}, confidence {effect.confidence:.2f}" if effect.enabled else "none found"

    print(f"  saturation: {state(s, f'drive {s.drive_db:.1f} dB, mix {s.mix_pct:.0f} %')}")
    print(f"  de-esser:   {state(d, f'{d.frequency_hz:.0f} Hz, thr {d.threshold_db:.1f} dB, up to {d.reduction_db:.1f} dB')}")
    print(f"  delay:      {state(dl, f'{dl.time_ms:.1f} ms, feedback {dl.feedback_pct:.0f} %, mix {dl.mix_pct:.0f} %, high-cut {dl.highcut_hz:.0f} Hz')}")
    print(f"  reverb:     {state(r, f'decay {r.decay_s:.2f} s, predelay {r.predelay_ms:.0f} ms, mix {r.mix_pct:.0f} %, damping {r.damping_hz:.0f} Hz')}")
    print(f"  doubler:    {state(w, f'mix {w.mix_pct:.0f} %, width {w.width_pct:.0f} %')}")
    pc = preset.pitch_correction
    if pc:
        tune = (f"{pc['amount_pct']:.0f} % (retune {pc['retune']}, ~{pc['retune_ms']:.0f} ms)" if pc["detected"]
                else f"none found ({pc['amount_pct']:.0f} %)")
        print(f"  pitch correction in the reference: {tune}; key {pc['key'] or '?'}, "
              f"tuning {pc['tuning_offset_cents']:+.0f} cents (not applied: v1 has no pitch correction)")


def per_singer_path(out: str | Path, singer: str) -> Path:
    """preset.json -> 'preset - Singer A.json' (next to it)."""
    out = Path(out)
    return out.with_name(f"{out.stem} - Singer {singer}{out.suffix or '.json'}")


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .audio_io import load_audio, write_audio
    from .chain import render
    from .pipeline import analyze
    from .selection import format_regions, parse_regions, select_reference
    from .separation import select_sections

    def progress(fraction: float, stage: str, lo: float = 0.0, hi: float = 1.0) -> None:
        # Machine-readable lines for the plug-in's Analyze button (--progress).
        if args.progress:
            print(f"PROGRESS {lo + (hi - lo) * fraction:.3f} {stage}", flush=True)

    try:
        regions = parse_regions(args.regions) if args.regions else None
    except ValueError as exc:
        return _fail(args, f"--regions: {exc}", 2)
    if args.per_singer and not args.singer:
        return _fail(args, "--per-singer needs --singer (e.g. --singer all)", 2)

    progress(0.0, "reading your dry vocal")
    dry, fs = load_audio(args.dry)
    stem, rest, separation, used_demucs = _load_reference_stem(args, fs, progress)
    sep_end = (0.05 if separation.get("cached") else 0.5) if used_demucs else 0.0
    base_mask = None
    if rest is not None:
        sel = select_sections(stem, rest, fs)
        base_mask = sel.mask
        separation.update(sel.diagnostics)
        if not regions and not args.singer:
            print(f"analysing {sel.kept_s:.1f} s of {sel.total_s:.1f} s (sections where the vocal is clearly present)")

    singer_map = None
    if args.singer:
        from .singers import detect_singers

        progress(0.0, "finding the singers", sep_end, sep_end + 0.05)
        singer_map = detect_singers(stem, fs)
        print(f"singers: {singer_map.message} (confidence {100 * singer_map.confidence:.0f} %)")
        try:
            chosen = _singer_ids(args.singer, singer_map)
        except ValueError as exc:
            return _fail(args, f"--singer: {exc}", 2)
        if singer_map.unsure:
            print("warning: singer detection is unsure; the result may mix singers. "
                  "Mark the sections to analyse with --regions instead.", file=sys.stderr)
    else:
        chosen = []

    # One analysis per singer with --per-singer, else one for the whole choice.
    jobs = [[c] for c in chosen] if args.per_singer else [chosen]
    lo = sep_end + (0.05 if chosen else 0.0)
    span = (1.0 - lo) / len(jobs)
    for j, who in enumerate(jobs):
        sel = select_reference(stem, fs, base_mask=base_mask, regions=regions, singer_map=singer_map, singers=who or None)
        for note in sel.notes:
            print(f"note: {note}", file=sys.stderr)
        if sel.analysed_s < 3.0:
            return _fail(args, f"Only {sel.analysed_s:.1f} s of vocal to analyse"
                  f"{' for singer ' + ', '.join(who) if who else ''}{' in the regions given' if regions else ''}: "
                  "choose more of the song.", 2)
        label = ("Singer " + " + ".join(who)) if who else ("regions " + format_regions(regions) if regions else "")
        what = f"analysing {sel.analysed_s:.1f} s of {sel.total_s:.1f} s"
        if who:
            what += f" (singer {', '.join(who)}; overlap excluded {sel.overlap_excluded_s:.1f} s)"
        elif regions:
            what += f" (regions {format_regions(regions)})"
        print(what)
        a = lo + j * span
        preset = analyze(sel.audio, dry, fs, reference_name=Path(args.reference).name, tempo_bpm=args.tempo,
                         passes=args.passes, refine_evals=args.refine,
                         progress=lambda f, st, a=a: progress(f, (label + ": " if len(jobs) > 1 else "") + st, a, a + span))
        if regions or who:
            preset.reference.selection = sel.to_dict()
        if separation:
            preset.diagnostics["separation"] = separation
        if singer_map is not None:
            preset.diagnostics["singers"] = {k: v for k, v in singer_map.to_dict().items() if k != "activity"}
        outs = ([Path(args.out)] if j == 0 else []) + ([per_singer_path(args.out, who[0])] if args.per_singer else [])
        for out in outs:
            preset.save(out)
            print(f"wrote {out}")
        if who:
            conf = f", singer confidence {100 * singer_map.confidence:.0f} %"
            print(f"{label}{conf}:")
        _print_preset(preset)
        if args.render and j == 0:
            write_audio(args.render, render(preset, dry, fs, tail_s=args.tail), fs)
            print(f"wrote {args.render}")
    print(_peak_memory())
    return 0


def _cmd_singers(args: argparse.Namespace) -> int:
    """Who sings when in the reference (for the plug-in's Analyze panel)."""
    import json

    from .singers import detect_singers

    def progress(fraction: float, stage: str, lo: float = 0.0, hi: float = 1.0) -> None:
        if args.progress:
            print(f"PROGRESS {lo + (hi - lo) * fraction:.3f} {stage}", flush=True)

    info = __import__("soundfile").info(str(args.reference))
    fs = int(info.samplerate)
    stem, _, separation, used_demucs = _load_reference_stem(args, fs, progress, span=0.85)
    progress(0.9 if used_demucs else 0.1, "finding the singers")
    m = detect_singers(stem, fs)
    d = {"version": 1, "reference": str(Path(args.reference).resolve()), "separated": used_demucs, **separation,
         **m.to_dict()}
    Path(args.json).write_text(json.dumps(d) + "\n", encoding="utf-8")
    progress(1.0, "done")
    print(m.message)
    for s in m.singers:
        print(f"  Singer {s.id}: {s.seconds:.1f} s, median pitch {s.median_f0_hz:.0f} Hz, preview {s.preview[0]:.1f}-{s.preview[1]:.1f} s")
    print(f"confidence {100 * m.confidence:.0f} %; wrote {args.json}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from .audio_io import load_audio, write_audio
    from .chain import render
    from .preset import Preset

    preset = Preset.load(args.preset)
    dry, fs = load_audio(args.dry)
    write_audio(args.out, render(preset, dry, fs, tail_s=args.tail, match_amount=args.match / 100.0), fs)
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
    from .selftest import run, run_separation
    from .separation import demucs_available

    print("Vocal Match self-test: synthetic vocal through a known EQ + compressor, then analysed (about a minute)")
    ok = run()
    if args.separation == "yes" or (args.separation == "auto" and demucs_available()):
        print("Source separation self-test: a synthetic song (vocal + drums, bass, chords) separated with Demucs "
              "(the first run downloads the model, about 80 MB)")
        ok = run_separation() and ok
    elif args.separation == "auto":
        print("  NOTE  source separation is not installed: give the analyzer isolated vocals, "
              "or run the installer again to add it")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vocalmatch", description="Estimate a vocal effect chain from a reference.")
    p.add_argument("--version", action="version", version=f"vocalmatch {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="estimate a preset from a reference vocal and your dry vocal")
    a.add_argument("--reference", required=True, help="reference song or vocal stem (WAV/FLAC/MP3)")
    a.add_argument("--dry", required=True, help="your dry vocal")
    a.add_argument("--out", required=True, help="output preset.json")
    a.add_argument("--render", help="also render the dry vocal through the preset to this file")
    a.add_argument("--tempo", type=float, default=None, help="song tempo in BPM (optional)")
    a.add_argument("--tail", type=float, default=0.0, help="seconds of tail to render after the dry ends")
    a.add_argument("--separate", choices=("auto", "yes", "no"), default="auto",
                   help="extract the vocal from a full song with Demucs: auto = when Demucs is installed (default); "
                        "no = --reference is already an isolated vocal stem")
    a.add_argument("--refine", type=int, default=60, metavar="N",
                   help="refinement optimizer evaluations after the estimators (about 1 s each; 0 = off; default 60)")
    a.add_argument("--passes", type=int, default=2, help="estimator passes over the chain (default 2)")
    a.add_argument("--regions", metavar="LIST",
                   help="analyse only these parts of the song: start-end in seconds or m:ss, comma-separated "
                        "(e.g. 12.5-30,1:05-1:20)")
    a.add_argument("--singer", metavar="IDS",
                   help="songs with more than one singer: analyse only these singers (A, B, ... as found by "
                        "'vocalmatch singers'; several: A,B; or all). Sections where singers overlap are excluded")
    a.add_argument("--per-singer", action="store_true",
                   help="with --singer: analyse each chosen singer on their own and save one preset per singer "
                        "('<out> - Singer A.json', ...); --out gets the first")
    a.add_argument("--progress", action="store_true", help="print 'PROGRESS <0-1> <stage>' lines (for the plug-in)")
    a.set_defaults(func=_cmd_analyze)

    g = sub.add_parser("singers", help="find the singers in a reference and when each one sings")
    g.add_argument("--reference", required=True, help="reference song or vocal stem")
    g.add_argument("--json", required=True, help="write the result here (singers, sections, vocal activity)")
    g.add_argument("--separate", choices=("auto", "yes", "no"), default="auto",
                   help="as for analyze: extract the vocal from a full song first")
    g.add_argument("--progress", action="store_true", help="print 'PROGRESS <0-1> <stage>' lines (for the plug-in)")
    g.set_defaults(func=_cmd_singers)

    r = sub.add_parser("render", help="process a dry vocal with a preset")
    r.add_argument("--preset", required=True)
    r.add_argument("--dry", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tail", type=float, default=0.0)
    r.add_argument("--match", type=float, default=100.0, help="the plug-in's Match amount, 0-100 %% (default 100)")
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
    t.add_argument("--separation", choices=("auto", "yes", "no"), default="auto",
                   help="also check source separation: auto = when Demucs is installed (default); yes = it must work")
    t.set_defaults(func=_cmd_selftest)
    return p


def _limit_threads() -> None:
    """Bounded native thread pools (OpenMP / MKL / OpenBLAS / Accelerate), before numpy loads.
    A plug-in host's environment can carry odd values; at most 4 threads keeps an 8 GB
    MacBook Air responsive while the host plays."""
    import os

    n = str(max(1, min(4, os.cpu_count() or 1)))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        v = os.environ.get(k, "")
        if not v.isdigit() or int(v) < 1:
            os.environ[k] = n


def _fail(args, message: str, code: int) -> int:
    """A failure the user can act on: 'error: ...' for Terminal and, for the plug-in
    (--progress), one 'ERROR <message>' line it shows as is."""
    print(f"error: {message}", file=sys.stderr, flush=True)
    if getattr(args, "progress", False):
        print("ERROR " + message.replace("\n", " "), flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    import faulthandler

    import signal

    try:
        faulthandler.enable(all_threads=True)  # a native crash prints where it happened
    except (AttributeError, ValueError, OSError):  # stderr without a file descriptor (tests)
        pass
    _limit_threads()
    if hasattr(signal, "SIGTERM"):  # the plug-in's Cancel: unwind, so the separation worker is stopped too
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    args = build_parser().parse_args(argv)
    from .separation import SeparationFailed, SeparationUnavailable

    try:
        return int(args.func(args))
    except SeparationUnavailable as exc:
        return _fail(args, str(exc), 2)
    except SeparationFailed as exc:
        return _fail(args, f"Source separation failed: {exc}", 3)
    except MemoryError:
        return _fail(args, "Not enough memory to analyse this song. Close other apps, or mark shorter regions "
                           "of the song to analyse, and try again.", 4)
    except FileNotFoundError as exc:
        return _fail(args, f"File not found: {exc.filename or exc}", 2)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - anything else: a clear line, the traceback in the log
        import traceback

        traceback.print_exc()
        name = type(exc).__name__
        if name == "LibsndfileError" or "Error opening" in str(exc):
            return _fail(args, f"Could not read an audio file ({str(exc).splitlines()[0]}). Use WAV, AIFF, FLAC or MP3.", 2)
        return _fail(args, f"The analyzer hit an internal error ({name}: {str(exc).splitlines()[0] if str(exc) else ''}).", 1)


if __name__ == "__main__":
    raise SystemExit(main())
