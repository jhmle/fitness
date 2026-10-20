import numpy as np
import pytest

import gen_data as g
from common import DATA_DIR, load_kb, load_rows, row_errors, validate


def test_phrasing_lists_are_held_out():
    for name in ["T_START", "T_REST_N", "T_REST_PLAIN", "T_SUMMARY", "T_RETRIEVE", "T_EASIER", "T_FEWER", "T_MORE",
                 "T_LONGER", "T_SHORTER", "T_TARGET", "T_SETREST"]:
        train, test = getattr(g, name)
        assert not set(train) & set(test), name
    for sec, (train, test) in g.DUR.items():
        assert not set(train) & set(test), sec


def _check_rows(rows, kb):
    assert rows
    for r in rows:
        assert validate(r["label"], r["allow"], row_errors(r)) is not None, r["id"]
        assert r["label"]["tool"] in r["allow"], r["id"]
        assert all(i in kb for i in r["knowledge"]) and len(r["knowledge"]) == len(set(r["knowledge"]))
        ex = r["state"]["exercise"]
        assert all(kb[i]["ex"] in (ex, "any") for i in r["knowledge"]), r["id"]
        if r["slice"] == "rep_cue":
            assert r["state"]["errors"] and r["state"]["rep"] < r["state"]["target"]
            assert len(r["label"]["args"]["text"].split()) <= 12
        if r["slice"] == "set_end":
            assert r["state"]["rep"] == r["state"]["target"]


def test_generator_produces_valid_labelled_rows():
    kb = load_kb()
    _check_rows(g.build_split(np.random.default_rng(1), kb, "val", 200), kb)


def test_teacher_policy_rules():
    kb = load_kb()
    st = {"errors": [{"id": "squat_depth", "sev": 2}, {"id": "squat_lean", "sev": 3}], "last_cue": None, "rest": 60, "target": 8, "trend": {"depth_loss_deg": 0}}
    assert g.label_rep(np.random.default_rng(0), kb, st)["args"]["cue_id"] == "squat_lean"          # highest severity
    st["last_cue"] = {"id": "squat_lean", "reps_ago": 1, "fixed": False}
    assert g.label_rep(np.random.default_rng(0), kb, st)["args"]["text"] in g.CUE_REPHRASE["squat_lean"]  # reworded after a miss
    assert g.label_set_end(st)["args"] == {"rest": 90, "target": 6}                                       # 2 errors -> adapt
    st["errors"] = []
    assert g.label_set_end(st)["args"] == {"rest": 60, "target": 8}                                       # clean set -> keep plan
    st["trend"]["depth_loss_deg"] = 13
    assert g.label_set_end(st)["args"] == {"rest": 90, "target": 6}                                       # fatigue -> adapt


def test_frozen_test_set_is_consistent():
    p = DATA_DIR / "test.jsonl"
    if not p.exists():
        pytest.skip("run gen_data.py first")
    rows = load_rows(p)
    _check_rows(rows, load_kb())
    assert len(rows) >= 300
