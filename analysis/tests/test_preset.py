import json
from pathlib import Path

import jsonschema
import pytest

from vocalmatch.preset import EFFECT_ORDER, Preset, load_schema, validate_preset_dict

REPO = Path(__file__).resolve().parents[2]


def test_default_preset_validates():
    validate_preset_dict(Preset().to_dict())


def test_json_round_trip(tmp_path):
    p = Preset()
    p.reference.filename = "song.wav"
    p.reference.tempo_bpm = 96.0
    p.effects.eq.enabled = True
    p.effects.eq.bands[2].gain_db = 3.5
    p.effects.eq.bands[2].enabled = True
    p.effects.compressor.ratio = 4.0
    p.effects.delay.sync_note = "1/8D"
    path = tmp_path / "p.json"
    p.save(path)
    q = Preset.load(path)
    assert q.to_dict() == p.to_dict()
    assert list(json.loads(path.read_text())["effects"]) == list(EFFECT_ORDER)


def test_shared_examples_validate():
    examples = sorted((REPO / "shared" / "examples").glob("*.json"))
    assert examples
    for path in examples:
        Preset.load(path)  # validates


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("schema_version", 2),
        lambda d: d["effects"]["eq"]["bands"][0].__setitem__("gain_db", 13.0),
        lambda d: d["effects"]["eq"]["bands"].append(dict(d["effects"]["eq"]["bands"][0])),
        lambda d: d["effects"]["compressor"].__setitem__("ratio", 0.5),
        lambda d: d["effects"]["reverb"].__setitem__("confidence", 1.5),
        lambda d: d["effects"]["delay"].__setitem__("sync_note", "dotted eighth"),
        lambda d: d["effects"]["doubler"].__setitem__("wet", 10),
        lambda d: d["effects"].pop("saturation"),
    ],
)
def test_schema_rejects_bad_presets(mutate):
    d = Preset().to_dict()
    mutate(d)
    with pytest.raises(jsonschema.ValidationError):
        validate_preset_dict(d)


def test_from_dict_rejects_unknown_version():
    d = Preset().to_dict()
    d["schema_version"] = 99
    with pytest.raises(ValueError):
        Preset.from_dict(d)


def test_schema_is_valid_draft_2020_12():
    jsonschema.Draft202012Validator.check_schema(load_schema())


def test_numpy_values_are_converted():
    import numpy as np

    p = Preset()
    p.effects.eq.bands[0].enabled = np.bool_(True)
    p.effects.eq.bands[0].gain_db = np.float64(2.0)
    p.diagnostics = {"curve": np.zeros(3)}
    d = p.to_dict()
    validate_preset_dict(d)
    assert type(d["effects"]["eq"]["bands"][0]["enabled"]) is bool
    assert d["diagnostics"]["curve"] == [0.0, 0.0, 0.0]
