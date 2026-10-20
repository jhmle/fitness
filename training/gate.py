"""Deployment gate. Exit code 1 blocks the GitHub Pages deploy.

  python gate.py --candidate metrics/candidate.json --production metrics/production.json --min-valid 0.95
  python gate.py --candidate metrics/candidate.json --production metrics/production.json --promote   # after it passes

Checks
  1. candidate was evaluated on the CURRENT frozen test set (hash match), otherwise the numbers are not comparable
  2. absolute floor on valid-JSON rate (--min-valid)
  3. no regression vs production on overall and per-slice action accuracy, advice-error rate, unsupported cues
  4. deploy consistency: if production.json names an ONNX model, config.js LLM_ID must be that exact model
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from common import APP_DIR, DATA_DIR, file_sha


def load(p):
    with open(p) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--production", required=True)
    ap.add_argument("--min-valid", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.01, help="allowed overall regression")
    ap.add_argument("--slice-tol", type=float, default=0.03, help="allowed per-slice regression")
    ap.add_argument("--promote", action="store_true", help="overwrite production.json with the candidate if all checks pass")
    ap.add_argument("--config", default=str(APP_DIR / "config.js"))
    a = ap.parse_args()

    for label, path in (("candidate", a.candidate), ("production", a.production)):
        if not pathlib.Path(path).exists():
            hint = ("it is written by the evaluation step: run `python eval.py --predictor hf --model <base> --adapter out/adapter --name candidate` first"
                    if label == "candidate" else "it is committed in the repo as training/metrics/production.json; run this from the training/ folder")
            print(f"{label} file not found: {path}\n  -> {hint}")
            return 2
    cand, prod = load(a.candidate), load(a.production)
    c, p = cand["metrics"], prod["metrics"]
    fails, rows = [], []

    def check(name, ok, detail):
        rows.append((("PASS" if ok else "FAIL"), name, detail))
        if not ok:
            fails.append(name)

    sha = file_sha(DATA_DIR / "test.jsonl")
    check("test-set hash", cand.get("test_sha") == sha, f"candidate {cand.get('test_sha')} vs current {sha}")
    check("valid-JSON floor", c["valid_json_rate"] >= a.min_valid, f"{c['valid_json_rate']:.3f} >= {a.min_valid}")
    check("action accuracy", c["action_acc"] >= p["action_acc"] - a.tol, f"{c['action_acc']:.3f} vs prod {p['action_acc']:.3f}")
    check("advice-error rate", c["advice_error_rate"] <= p["advice_error_rate"] + a.tol, f"{c['advice_error_rate']:.3f} vs prod {p['advice_error_rate']:.3f}")
    check("unsupported cues", c["unsupported_cue_rate"] <= p["unsupported_cue_rate"] + a.tol, f"{c['unsupported_cue_rate']:.3f} vs prod {p['unsupported_cue_rate']:.3f}")
    for k in sorted(k for k in p if k.endswith("_action_acc") and k != "action_acc"):
        check(f"slice {k}", c.get(k, 0) >= p[k] - a.slice_tol, f"{c.get(k, 0):.3f} vs prod {p[k]:.3f}")

    onnx = prod.get("onnx_model")
    try:
        m = re.search(r'LLM_ID\s*=\s*"([^"]+)"', open(a.config, encoding="utf-8").read())
        deployed = m.group(1) if m else None
    except FileNotFoundError:
        deployed = None
    if onnx:
        check("deploy consistency", deployed == onnx, f"config.js LLM_ID={deployed} vs production {onnx}")

    for status, name, detail in rows:
        print(f"[{status}] {name}: {detail}")
    if fails:
        print(f"\nGATE FAILED ({len(fails)}): " + ", ".join(fails))
        return 1
    print("\nGATE PASSED")
    if a.promote:
        with open(a.production, "w") as f:
            json.dump(cand, f, indent=1)
        print("promoted candidate -> production")
    return 0


if __name__ == "__main__":
    sys.exit(main())
