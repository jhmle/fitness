"""Simple deploy check: ship the new model if it matches or beats the old one on ANY measure.

  python should_deploy.py              # compare only (exit 0 = deploy, exit 1 = hold back)
  python should_deploy.py --promote    # compare, and if it passes: make the new model the live one

The rule
  - Better than, or equal to, the old model on at least one measure -> deploy
  - Worse on every single measure                                   -> don't deploy (the old model stays live)

Safety checks (the only two): both models must have been scored on the same test set, and the new
model must have been exported (so the app has something to load).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from common import APP_DIR, DATA_DIR, METRICS_DIR, file_sha

HIGHER_IS_BETTER = {"valid_json_rate", "tool_acc", "worded_cue_rate", "rephrase_rate"}
LOWER_IS_BETTER = {"unsupported_cue_rate"}
# advice_error_rate and invalid_rate are just 1 minus action_acc / valid_json_rate, so they are not counted twice.


def direction(key: str):
    """+1 if bigger is better, -1 if smaller is better, None if the metric is not compared."""
    if key in HIGHER_IS_BETTER or key.endswith("action_acc") or key.endswith("_valid_rate"):
        return 1
    if key in LOWER_IS_BETTER:
        return -1
    return None


def compare(new: dict, old: dict):
    rows = []
    for key in sorted(set(new) & set(old)):
        d = direction(key)
        a, b = new[key], old[key]
        if d is None or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        diff = (a - b) * d
        rows.append((key, b, a, "better" if diff > 1e-9 else "worse" if diff < -1e-9 else "same"))
    return rows


def promote(cand: dict, cand_path: pathlib.Path, prod_path: pathlib.Path, config_path: pathlib.Path) -> None:
    """New model becomes production: save its scores and point the app (config.js) at it."""
    text = config_path.read_text(encoding="utf-8")
    model = cand["onnx_model"]
    text, n = re.subn(r'(LLM_ID\s*=\s*")[^"]+(")', lambda m: m.group(1) + model + m.group(2), text, count=1)
    if n != 1:
        sys.exit(f"Could not find LLM_ID in {config_path}")
    config_path.write_text(text, encoding="utf-8")
    prod_path.write_text(json.dumps(cand, indent=1))
    cand_path.unlink()
    print(f"Promoted: the app now uses {model}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=str(METRICS_DIR / "candidate.json"))
    ap.add_argument("--production", default=str(METRICS_DIR / "production.json"))
    ap.add_argument("--config", default=str(APP_DIR / "config.js"))
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    cand_path, prod_path = pathlib.Path(a.candidate), pathlib.Path(a.production)
    if not cand_path.exists():
        print(f"No new model to check ({cand_path} not found).")
        return 2
    cand = json.loads(cand_path.read_text())

    if not cand.get("onnx_model"):
        print("HOLD BACK: the new model was never exported (no onnx_model in its results).")
        return 1
    if cand.get("test_sha") != file_sha(DATA_DIR / "test.jsonl"):
        print("HOLD BACK: the new model was scored on a different test set, so the results can't be compared.")
        return 1

    if not prod_path.exists():
        print("No current model to compare against: deploying the new one.")
        passing = 1
    else:
        rows = compare(cand["metrics"], json.loads(prod_path.read_text())["metrics"])
        print(f"{'measure':28} {'old':>8} {'new':>8}")
        for key, old, new, status in rows:
            print(f"{key:28} {old:8.3f} {new:8.3f}  {status}")
        better = sum(r[3] == "better" for r in rows)
        worse = sum(r[3] == "worse" for r in rows)
        same = len(rows) - better - worse
        print(f"\n{better} better, {same} unchanged, {worse} worse")
        passing = better + same          # "greater than or equal to" the old model

    if not passing:
        print("\nHOLD BACK: the new model is worse on every measure. The current model stays live.")
        return 1
    print("\nDEPLOY: the new model is at least as good as the current one on at least one measure.")
    if a.promote:
        promote(cand, cand_path, prod_path, pathlib.Path(a.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
