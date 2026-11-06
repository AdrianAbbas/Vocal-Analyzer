"""Preset data model, mirroring /shared/preset.schema.json.

Every effect carries ``enabled`` and ``confidence`` (0..1) plus its settings in
real units. Key names carry the unit suffix (``_hz``, ``_db``, ``_ms``, ``_s``,
``_pct``); ``ratio`` and ``q`` are unitless.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from . import __version__

SCHEMA_VERSION = 1

EFFECT_ORDER = (
    "input_gain",
    "eq",
    "compressor",
    "saturation",
    "deesser",
    "delay",
    "reverb",
    "doubler",
    "output_gain",
)

EQ_BAND_TYPES = ("low_shelf", "peak", "high_shelf")
MAX_EQ_BANDS = 6


@dataclass
class Effect:
    enabled: bool = False
    confidence: float = 0.0


@dataclass
class Gain(Effect):
    gain_db: float = 0.0


@dataclass
class EQBand:
    type: str = "peak"
    freq_hz: float = 1000.0
    gain_db: float = 0.0
    q: float = 0.7071
    enabled: bool = True


def default_eq_bands() -> list[EQBand]:
    """The plugin's fixed 6-band layout: low shelf, 4 peaks, high shelf."""
    return [
        EQBand("low_shelf", 100.0, 0.0, 0.7071, False),
        EQBand("peak", 250.0, 0.0, 1.0, False),
        EQBand("peak", 800.0, 0.0, 1.0, False),
        EQBand("peak", 2500.0, 0.0, 1.0, False),
        EQBand("peak", 5000.0, 0.0, 1.0, False),
        EQBand("high_shelf", 10000.0, 0.0, 0.7071, False),
    ]


@dataclass
class EQ(Effect):
    bands: list[EQBand] = field(default_factory=default_eq_bands)


@dataclass
class Compressor(Effect):
    threshold_db: float = 0.0
    ratio: float = 1.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 6.0
    makeup_db: float = 0.0


@dataclass
class Saturation(Effect):
    drive_db: float = 0.0
    mix_pct: float = 0.0


@dataclass
class DeEsser(Effect):
    frequency_hz: float = 6500.0
    threshold_db: float = 0.0
    reduction_db: float = 0.0


@dataclass
class Delay(Effect):
    time_ms: float = 250.0
    feedback_pct: float = 0.0
    mix_pct: float = 0.0
    highcut_hz: float = 8000.0
    sync_note: str | None = None


@dataclass
class Reverb(Effect):
    decay_s: float = 1.5
    predelay_ms: float = 0.0
    mix_pct: float = 0.0
    damping_hz: float = 8000.0


@dataclass
class Doubler(Effect):
    mix_pct: float = 0.0
    width_pct: float = 100.0


@dataclass
class Effects:
    input_gain: Gain = field(default_factory=Gain)
    eq: EQ = field(default_factory=EQ)
    compressor: Compressor = field(default_factory=Compressor)
    saturation: Saturation = field(default_factory=Saturation)
    deesser: DeEsser = field(default_factory=DeEsser)
    delay: Delay = field(default_factory=Delay)
    reverb: Reverb = field(default_factory=Reverb)
    doubler: Doubler = field(default_factory=Doubler)
    output_gain: Gain = field(default_factory=Gain)


@dataclass
class Reference:
    filename: str = ""
    sample_rate_hz: int = 48000
    tempo_bpm: float | None = None
    duration_s: float | None = None


@dataclass
class Preset:
    reference: Reference = field(default_factory=Reference)
    effects: Effects = field(default_factory=Effects)
    schema_version: int = SCHEMA_VERSION
    generator: dict[str, str] = field(
        default_factory=lambda: {"name": "vocalmatch-analysis", "version": __version__}
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "generator": dict(self.generator),
            "reference": asdict(self.reference),
            "effects": {name: asdict(getattr(self.effects, name)) for name in EFFECT_ORDER},
        }
        if self.diagnostics:
            d["diagnostics"] = self.diagnostics
        return _plain(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Preset":
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported preset schema_version {version!r} (expected {SCHEMA_VERSION})")
        effects_in = d.get("effects", {})
        effects = Effects()
        for f in fields(Effects):
            if f.name not in effects_in:
                continue
            raw = dict(effects_in[f.name])
            current = getattr(effects, f.name)
            if f.name == "eq":
                raw["bands"] = [EQBand(**b) for b in raw.get("bands", [])]
            setattr(effects, f.name, type(current)(**raw))
        return cls(
            reference=Reference(**d.get("reference", {})),
            effects=effects,
            schema_version=version,
            generator=dict(d.get("generator", {})),
            diagnostics=dict(d.get("diagnostics", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Preset":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | os.PathLike, validate: bool = True) -> None:
        d = json.loads(self.to_json())
        if validate:
            validate_preset_dict(d)
        Path(path).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike, validate: bool = True) -> "Preset":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if validate:
            validate_preset_dict(d)
        return cls.from_dict(d)


def _plain(o: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        return o.tolist()
    return o


def _json_default(o: Any) -> Any:
    # numpy scalars / arrays sneak into diagnostics
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------- validation
def schema_path() -> Path:
    env = os.environ.get("VOCALMATCH_SCHEMA")
    if env:
        return Path(env)
    # analysis/vocalmatch/preset.py -> repo root
    return Path(__file__).resolve().parents[2] / "shared" / "preset.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def validate_preset_dict(d: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if *d* does not follow the schema."""
    import jsonschema

    jsonschema.validate(instance=d, schema=load_schema(), cls=jsonschema.Draft202012Validator)
