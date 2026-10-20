"""Evaluate a coach-brain predictor on the frozen test set.

  python eval.py --predictor rules --name rules                         # the shipped rule-based policy (no GPU)
  python eval.py --predictor hf --model HuggingFaceTB/SmolLM2-360M-Instruct --name base
  python eval.py --predictor hf --model HuggingFaceTB/SmolLM2-360M-Instruct --adapter out/adapter --name candidate \
                 --register trainer-coach

Every prediction goes through the SAME validate() the browser uses, so "valid" means "the agent would have accepted it".
Metrics are written to metrics/<name>.json (what CI gates on) and logged to MLflow (params, metrics, confusion matrix,
adapter artifact, git SHA, test-set hash) when mlflow is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix

from common import (DATA_DIR, METRICS_DIR, ROOT, extract_call, file_sha, load_kb, load_rows, load_system, row_errors,
                    to_messages, validate)

TOOLS = ["give_cue", "adjust_plan", "retrieve_cues", "rest_timer", "start_exercise", "set_summary", "invalid"]


# ---------------------------------------------------------------- rules baseline (port of TemplatePolicy + regexIntent)
def predict_rules(row: dict) -> dict | None:
    st = row["state"]
    if row["kind"] == "REP":
        if st["rep"] >= st["target"]:
            # TemplatePolicy never adapts the plan from the fatigue trend: its set-end decision is "keep the plan".
            return {"tool": "adjust_plan", "args": {"rest": st["rest"], "target": st["target"]}}
        top = sorted(st["errors"], key=lambda e: -e["sev"])[0]
        return {"tool": "give_cue", "args": {"cue_id": top["id"], "tone": "firm"}}
    t = row["said"].lower()
    if re.search(r"start|begin|switch|let's do", t) and re.search(r"squat|push|curl", t):
        name = "squat" if "squat" in t else "pushup" if "push" in t else "curl"
        return {"tool": "start_exercise", "args": {"name": name}}
    if re.search(r"how (was|did)|summary|finish|done|end (the )?set", t):
        return {"tool": "set_summary", "args": {}}
    if re.search(r"rest|break", t):
        return {"tool": "rest_timer", "args": {"seconds": st["rest"]}}
    if re.search(r"\b(why|how|what|should|tips?|help)\b", t):
        return {"tool": "retrieve_cues", "args": {"query": row["said"]}}
    return None


def run_rules(rows):
    return [{"raw": None, "call": predict_rules(r)} for r in rows]


# ---------------------------------------------------------------- HF model predictor (base or base + LoRA adapter)
def make_hf_runner(model_id: str, adapter: str | None, batch_size: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import transformers   # `torch_dtype` was renamed to `dtype` in transformers 4.56; pick whichever this install understands
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
    dtype_kw = {("dtype" if ver >= (4, 56) else "torch_dtype"): torch.float16 if dev == "cuda" else torch.float32}
    model = AutoModelForCausalLM.from_pretrained(model_id, **dtype_kw).to(dev)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    kb, system = load_kb(), load_system()

    def run(rows):
        outs = []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = [tok.apply_chat_template(to_messages(r, kb, system), tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():   # same decoding as llm.worker.js: greedy, max 70 new tokens
                g = model.generate(**enc, max_new_tokens=70, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outs += [{"raw": t, "call": extract_call(t)} for t in texts]
            print(f"  {min(i + batch_size, len(rows))}/{len(rows)}", end="\r", flush=True)
        print()
        return outs
    return run


# ---------------------------------------------------------------- scoring
def action_match(p: dict, y: dict) -> bool:
    """Same tool AND the arguments that decide behaviour (free-text query/text are not compared)."""
    if p["tool"] != y["tool"]:
        return False
    a, b = p["args"], y["args"]
    t = y["tool"]
    if t == "give_cue":
        return a["cue_id"] == b["cue_id"]
    if t == "start_exercise":
        return a["name"] == b["name"]
    if t == "rest_timer":
        return a["seconds"] == b["seconds"]
    if t == "adjust_plan":
        return a["rest"] == b["rest"] and a["target"] == b["target"]
    return True


def norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def score(rows, outs, kb) -> dict:
    n = len(rows)
    y_true, y_pred, per = [], [], {}
    worded = rephrase_n = rephrase_ok = unsupported = rep_n = 0
    for r, o in zip(rows, outs):
        errs, lab = row_errors(r), r["label"]
        raw = o["call"]
        v = validate(raw, r["allow"], errs)
        ok_action = bool(v and action_match(v, lab))
        s = per.setdefault(r["slice"], Counter())
        s["n"] += 1; s["valid"] += bool(v); s["action"] += ok_action; s["tool"] += bool(v and v["tool"] == lab["tool"])
        y_true.append(lab["tool"]); y_pred.append(v["tool"] if v else "invalid")
        if r["slice"] == "rep_cue":
            rep_n += 1
            if isinstance(raw, dict) and raw.get("tool") == "give_cue" and (raw.get("args") or {}).get("cue_id") not in errs:
                unsupported += 1          # model tried to cue something that did not happen
            txt = (v or {}).get("args", {}).get("text", "") if v and v["tool"] == "give_cue" else ""
            worded += bool(txt) and len(txt.split()) <= 12
            lc = r["state"]["last_cue"]
            if lc and lc["fixed"] is False and lc["id"] == lab["args"]["cue_id"]:
                rephrase_n += 1
                rephrase_ok += bool(txt) and norm(txt) != norm(kb[lc["id"]]["cue"])
    tot = lambda k: sum(s[k] for s in per.values())  # noqa: E731
    m = {"n": n, "valid_json_rate": tot("valid") / n, "tool_acc": tot("tool") / n, "action_acc": tot("action") / n}
    m["advice_error_rate"] = 1 - m["action_acc"]
    m["invalid_rate"] = 1 - m["valid_json_rate"]
    m["unsupported_cue_rate"] = unsupported / max(1, rep_n)
    m["worded_cue_rate"] = worded / max(1, rep_n)
    m["rephrase_rate"] = rephrase_ok / rephrase_n if rephrase_n else None
    for k, s in per.items():
        m[f"{k}_action_acc"], m[f"{k}_valid_rate"] = s["action"] / s["n"], s["valid"] / s["n"]
    m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    rep = classification_report(y_true, y_pred, labels=[t for t in TOOLS if t in set(y_true) | set(y_pred)], output_dict=True, zero_division=0)
    labels = [t for t in TOOLS if t in set(y_true) | set(y_pred)]
    return {"metrics": m, "tool_report": rep, "confusion": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}}


# ---------------------------------------------------------------- tracking
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def log_mlflow(a, result, adapter):
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed: skipping tracking (pip install mlflow)")
        return
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.name) as run:
        mlflow.log_params({"predictor": a.predictor, "model": a.model or "-", "adapter": adapter or "-", "n_test": result["n"], "limit": a.limit or 0})
        mlflow.set_tags({"git_sha": result["git_sha"], "test_sha": result["test_sha"], "stage": "evaluated"})
        mlflow.log_metrics({k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))})
        mlflow.log_dict(result, "eval.json")
        if adapter and os.path.isdir(adapter):
            mlflow.log_artifacts(adapter, "adapter")
            if a.register:
                try:
                    # MLflow 3 only lets runs:/ URIs point at "logged models", and a plain artifact folder isn't one.
                    # Registering from the artifact location works on MLflow 2 and 3 (and needs no flavor).
                    from mlflow import MlflowClient
                    client = MlflowClient()
                    try:
                        client.create_registered_model(a.register)
                    except Exception:   # already exists
                        pass
                    mv = client.create_model_version(a.register, source=f"{run.info.artifact_uri}/adapter", run_id=run.info.run_id)
                    print(f"registered {a.register} v{mv.version}")
                except Exception as e:   # registry support varies by backend/version; metrics are already saved
                    print("model registry step skipped:", e)
        print("MLflow run:", run.info.run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", choices=["rules", "hf"], required=True)
    ap.add_argument("--model", help="HF model id or local path (hf predictor)")
    ap.add_argument("--adapter", help="LoRA adapter dir from finetune_qlora.py")
    ap.add_argument("--name", required=True, help="writes metrics/<name>.json")
    ap.add_argument("--data", default=str(DATA_DIR / "test.jsonl"))
    ap.add_argument("--out", help="override output path")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (CI smoke runs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--experiment", default="ai-trainer-coach")
    ap.add_argument("--register", help="MLflow registered-model name (needs --adapter)")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    rows = load_rows(a.data)
    if a.limit:
        rows = rows[:a.limit]
    kb = load_kb()
    t0 = time.time()
    if a.predictor == "rules":
        outs = run_rules(rows)
    else:
        if not a.model:
            ap.error("--model is required for --predictor hf")
        outs = make_hf_runner(a.model, a.adapter, a.batch_size)(rows)
    result = score(rows, outs, kb)
    result.update(name=a.name, predictor=a.predictor, model=a.model, adapter=a.adapter, n=len(rows), git_sha=git_sha(),
                  test_sha=file_sha(a.data), seconds=round(time.time() - t0, 1))
    out = a.out or METRICS_DIR / f"{a.name}.json"
    METRICS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["metrics"], indent=1))
    print("wrote", out)
    if not a.no_mlflow:
        log_mlflow(a, result, a.adapter)


if __name__ == "__main__":
    main()
