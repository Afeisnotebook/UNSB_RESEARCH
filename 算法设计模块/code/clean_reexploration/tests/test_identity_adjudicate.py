import json

from clean_reexploration.adjudicate import adjudicate
from clean_reexploration.identity import canonical_json, sha256_bytes


def test_canonical_json_is_stable_across_dict_order():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_bytes(canonical_json(a).encode()) == sha256_bytes(canonical_json(b).encode())


def test_adjudicate_development_gain():
    ev = {
        "run_id": "r",
        "canonical_plain": {"psnr_macro": 18.0},
        "dt": {
            "delta_psnr": 0.2,
            "delta_psnr_ci_low": 0.05,
            "positive_domains": 4,
        },
    }
    out = adjudicate(ev)
    assert out["labels"]["dt"] == "DEVELOPMENT_GAIN"


def test_adjudicate_no_gain_when_ci_includes_zero():
    ev = {
        "run_id": "r",
        "canonical_plain": {"psnr_macro": 18.0},
        "dt": {
            "delta_psnr": 0.2,
            "delta_psnr_ci_low": -0.01,
            "positive_domains": 4,
        },
    }
    assert adjudicate(ev)["labels"]["dt"] == "DEVELOPMENT_NO_GAIN"


def test_adjudicate_handoff_gain():
    ev = {
        "run_id": "r",
        "canonical_plain": {"psnr_macro": 18.0},
        "hnek_full": {"psnr_macro": 18.8},
        "hnek_handoff": {
            "psnr_macro": 19.0,
            "vs_full_ci_low": 0.05,
            "vs_plain_ci_low": 0.05,
            "positive_domains_vs_plain": 4,
        },
    }
    assert adjudicate(ev)["labels"]["hnek_handoff_vs_full"] == "HANDOFF_OPTIMIZATION_GAIN"
