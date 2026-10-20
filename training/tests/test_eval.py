import json

from common import DATA_DIR, METRICS_DIR, load_kb, load_rows
from eval import run_rules, score


def test_rules_baseline_is_valid_on_rep_and_set_end_rows():
    rows = [r for r in load_rows(DATA_DIR / "test.jsonl") if r["slice"] != "voice"]
    m = score(rows, run_rules(rows), load_kb())["metrics"]
    assert m["valid_json_rate"] == 1.0 and m["rep_cue_action_acc"] == 1.0


def test_rules_metrics_do_not_regress_vs_committed_baseline():
    rows = load_rows(DATA_DIR / "test.jsonl")
    m = score(rows, run_rules(rows), load_kb())["metrics"]
    base = json.load(open(METRICS_DIR / "rules.json"))["metrics"]
    assert m["action_acc"] >= base["action_acc"] - 1e-6


def test_perfect_predictor_scores_one_and_garbage_scores_zero():
    rows = load_rows(DATA_DIR / "test.jsonl")[:80]
    kb = load_kb()
    perfect = score(rows, [{"raw": None, "call": r["label"]} for r in rows], kb)["metrics"]
    assert perfect["action_acc"] == 1.0 and perfect["valid_json_rate"] == 1.0 and perfect["advice_error_rate"] == 0.0
    garbage = score(rows, [{"raw": "nope", "call": None} for r in rows], kb)["metrics"]
    assert garbage["action_acc"] == 0.0 and garbage["invalid_rate"] == 1.0
    wrong_cue = [{"raw": None, "call": {"tool": "give_cue", "args": {"cue_id": "curl_swing", "text": "x"}}} for r in rows]
    assert score(rows, wrong_cue, kb)["metrics"]["unsupported_cue_rate"] > 0.5
