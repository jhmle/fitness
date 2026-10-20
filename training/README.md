# training/: fine-tune, evaluate, gate, deploy the coach brain

The browser agent (`web/agent.js`) asks a small LLM for **one JSON tool call** on rep errors, set ends and voice commands.
This folder fine-tunes that model with **QLoRA**, scores it against the rule-based policy on a **frozen test set**, tracks
everything in **MLflow**, and lets **GitHub Actions** block a deploy that regresses.

```
web/agent.js ──(SYSTEM prompt + KB parsed at runtime)──► common.py ──► gen_data.py ──► data/{train,val,test}.jsonl
                                                                                           │
                       Kaggle GPU:  finetune_qlora.py ──► out/adapter ──► eval.py ──► metrics/candidate.json
                                                                              │ (MLflow: params, metrics, adapter, version)
GitHub Actions:  pytest ► eval rules ► gate.py (vs metrics/production.json) ► deploy web/ to Pages
```

| File | What it does |
|---|---|
| `common.py` | Parses `SYSTEM` + `KB` from `agent.js` (single source of truth), builds prompts, Python port of `validate()` |
| `gen_data.py` | NumPy rep simulator (thresholds from `exercises.js`, fatigue drift) + teacher policy → labelled decisions. Test voice phrasings are held out from train |
| `finetune_qlora.py` | 4-bit NF4 + LoRA (PyTorch/PEFT/bitsandbytes), loss only on the JSON answer, logs to MLflow |
| `eval.py` | Rules baseline or HF model (+adapter). scikit-learn metrics, confusion matrix, MLflow logging/registration |
| `gate.py` | Regression gate + test-set hash check + "deployed model == evaluated model" check |
| `merge_export.py` | Merge adapter into fp16 base, optional push to HF Hub (for ONNX conversion) |
| `tests/` | 14 tests incl. a Node test that runs the **real** `agent.js` and asserts the training prompt is byte-identical |
| `notebooks/kaggle_train.ipynb` | Run-all Kaggle notebook |

Metrics (all on the frozen test set, every output goes through the same `validate()` the browser uses):
`valid_json_rate`, `action_acc` (right tool **and** right cue_id / exercise / seconds / plan numbers), `advice_error_rate` (= 1 − action_acc),
`unsupported_cue_rate` (model cued an error that didn't happen), per-slice accuracy (`rep_cue`, `set_end`, `voice`), `rephrase_rate`.

---

## 0. Put it in your repo (once)

Repo layout expected: `web/` (your app), `training/` (this folder), `.github/workflows/ci.yml`.
Copy `training/` and `.github/` into the repo root, then:

```bash
cd training
pip install -r requirements-ci.txt          # numpy, scikit-learn, mlflow, pytest
python gen_data.py                           # writes data/train|val|test.jsonl (test is frozen: never overwritten)
pytest tests -q                              # 14 tests; the Node parity test needs `node` installed
python eval.py --predictor rules --name rules      # baseline numbers, ~1 s
```
Commit `data/test.jsonl`, `metrics/rules.json`, `metrics/production.json` (already included). `train.jsonl`/`val.jsonl` are git-ignored; regenerate any time.

## 1. Train on Kaggle (free)

1. Push the repo to GitHub (public, or add a token to the clone URL).
2. kaggle.com → **Create → New Notebook** → *File → Import Notebook* → upload `training/notebooks/kaggle_train.ipynb`.
3. Right sidebar → **Session options**: Accelerator **GPU T4 x2** (or P100), **Internet: On** (phone verification required).
4. Edit `REPO_URL` in the first cell → **Run All** (~15–30 min).
5. From the **Output** tab download `results.zip`: contains `training/metrics/{base,candidate}.json`, `training/out/adapter`, `mlruns/`.

Useful knobs: `python finetune_qlora.py --epochs 3 --lora-r 32 --lr 1e-4`. Mix in real data: download **Download session log** from the app after a few
workouts, then `python gen_data.py --logs ~/Downloads/trainer-session*.json` (real states are relabelled by the teacher policy and added to train only).

## 2. Look at the MLflow runs

```bash
unzip results.zip -d .            # restores mlruns/ at the repo root
pip install mlflow && mlflow ui --backend-store-uri ./mlruns     # http://127.0.0.1:5000
```
You'll see a *trained* run (loss curves, hyper-parameters, adapter), an *evaluated* run per model (metrics, `eval.json` with the confusion matrix) and the
registered model `trainer-coach` (a new version per `--register` run). Runs are tagged with `git_sha` and `test_sha`. Want a hosted UI? Create a free DagsHub repo and set
`MLFLOW_TRACKING_URI` + `MLFLOW_TRACKING_USERNAME/PASSWORD` in the notebook.

## 3. Gate it

```bash
# results.zip contains training/metrics/candidate.json: copy it into your repo's training/metrics/ and commit it
python gate.py --candidate metrics/candidate.json --production metrics/production.json --min-valid 0.95
```
Prints PASS/FAIL per check. `production.json` starts as the **rules baseline**, so a candidate must match rules on `rep_cue` (cue choice is deterministic)
and beat it on `voice`/`set_end`. If your first model narrowly misses one slice, retrain with more epochs or loosen `--slice-tol`; don't edit the metrics.
When it passes and you ship it: `python gate.py ... --promote` makes it the new production baseline.

## 4. Ship it to the browser (ONNX for transformers.js)

```bash
HF_TOKEN=... python merge_export.py --adapter out/adapter --out out/merged --push YOUR_HF_USER/trainer-coach-smollm2-360m   # (notebook step 6)

# On any machine with Python (Kaggle works too): convert to ONNX + quantise with the transformers.js script.
git clone https://github.com/huggingface/transformers.js && cd transformers.js
pip install -r scripts/requirements.txt
python -m scripts.convert --quantize --model_id YOUR_HF_USER/trainer-coach-smollm2-360m      # check `--help` for current flags
# upload the produced folder (must contain onnx/model_q4.onnx) to a NEW HF repo, e.g. YOUR_HF_USER/trainer-coach-smollm2-360m-onnx
```
Then: set `LLM_ID = "YOUR_HF_USER/trainer-coach-smollm2-360m-onnx"` in `web/config.js`, add `"onnx_model": "<same id>"` to `metrics/candidate.json`, run `gate.py --promote`.
From then on CI fails if `config.js` points at a different model than the one that was evaluated.

> The conversion step and the in-browser q4 quantisation are the least-tested links. `eval.py` scores the **fp16 merged** model; the browser runs **q4 ONNX**.
> After deploying, use "Type a command instead" and the app's *Cue fix rate* line to sanity-check real behaviour.

## 5. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` on every push/PR: `pytest` → evaluate rules (logged to MLflow, uploaded as artifact) → gate rules vs `metrics/rules.json` →
(if `metrics/candidate.json` exists) gate candidate vs production → on `main` deploy `web/` to GitHub Pages.
Manual run (**Actions → ci → Run workflow**) with `model_id` re-computes a model's metrics on CPU (first 150 test rows) and gates them, so committed numbers can be spot-checked.

One-time setting: repo **Settings → Pages → Source: GitHub Actions**. If you deploy another way (branch/root), delete the `deploy` job and make the
`test-and-gate` job a required status check instead.

## Be honest about what this measures

* Training data is **synthetic** (simulated reps + a teacher policy that encodes the rules in your SYSTEM prompt) plus optional real logs. The model learns to follow
  those rules and to paraphrase cues; it is not trained on expert coach judgement.
* The rules baseline scores low on `voice` partly by construction: the test uses held-out paraphrases ("hang on i'm tired") that your regexes don't cover.
  That's the real gap an LLM closes, but quote the per-slice numbers, not just the headline.
* "Advice error" here means *wrong action for the state* (cue for an error that didn't happen, wrong plan change, wrong exercise), not medical correctness.
