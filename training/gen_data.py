"""Generate training / validation / frozen-test data for the coach-brain LLM.

Each row is one decision the agent asks the LLM to make (agent.js ask()):
  REP events with errors      -> give_cue
  Set end                     -> adjust_plan
  VOICE commands              -> retrieve_cues / rest_timer / start_exercise / set_summary / adjust_plan

Rep events are simulated with NumPy using the same thresholds as exercises.js, with fatigue drift so the
`trend.depth_loss_deg` feature is realistic. Labels come from a deterministic teacher policy that follows the
rules written in the SYSTEM prompt. Test voice phrasings are HELD OUT from the train phrasings, so the
test set measures generalisation, not memorisation.

Usage:
  python gen_data.py                                   # train/val (+ test if missing)
  python gen_data.py --logs ~/Downloads/trainer-session*.json   # also mix in real states from your sessions
  python gen_data.py --refresh-test                    # overwrite the frozen test set (invalidates old metrics!)
"""
from __future__ import annotations

import argparse
import glob
import json
import math

import numpy as np

from common import ALLOW_REP, ALLOW_SET_END, ALLOW_VOICE, DATA_DIR, NAMES, file_sha, js_dumps, load_kb

# ---- thresholds copied from exercises.js check(); depth = min joint angle (degrees) above which depth fails.
SIM = {
    #          depth_id        form_id        base_min  sd   depth_thr  form(mean,sd,thr,'gt'|'lt')   fast_thr  dur(mean,sd)
    "squat":  ("squat_depth",  "squat_lean",  98, 14, 110, (45, 13, 60, "gt"),  1.2, (1.9, 0.5)),
    "pushup": ("pushup_depth", "pushup_sag",  85, 14, 100, (168, 7, 160, "lt"), 0.8, (1.5, 0.4)),
    "curl":   ("curl_range",   "curl_swing",  55, 12, 70,  (10, 6, 15, "gt"),   0.8, (1.5, 0.4)),
}

CUE_FIRST = {
    "squat_depth":  ["Go a bit deeper.", "Sink a little lower.", "Drop your hips lower.", "Try to get deeper on this one."],
    "squat_lean":   ["Chest up.", "Chest up and brace your core.", "Keep your chest tall.", "Stay tall through your chest."],
    "pushup_depth": ["Lower your chest more.", "Go a little lower.", "Chest closer to the floor.", "Bend your elbows more."],
    "pushup_sag":   ["Hips up, stay straight.", "Keep your hips up.", "Tighten your core, stay straight.", "Don't let your hips sag."],
    "curl_range":   ["Curl all the way up.", "Bring the weight up higher.", "Finish the curl at the top.", "Full range, curl higher."],
    "curl_swing":   ["Keep your torso still.", "Stop swinging.", "Stay tall, no swinging.", "Keep your body still."],
    "too_fast":     ["Slow down, control the lowering.", "Easy tempo on the way down.", "Slow the lowering phase.", "Control it on the way down."],
}
CUE_REPHRASE = {  # used when the previous cue for the same error did NOT fix it: say it differently
    "squat_depth":  ["Sit your hips back and down.", "Push hips back, thighs near parallel.", "Hips back and down until thighs are parallel."],
    "squat_lean":   ["Core tight, chest tall, eyes forward.", "Brace your core and sit down tall.", "Sit down while keeping your chest proud."],
    "pushup_depth": ["Go down until your elbows hit ninety degrees.", "Bend elbows to ninety before pushing up.", "Lower until your elbows are square."],
    "pushup_sag":   ["Squeeze glutes and abs, stay planked.", "Keep your body in one straight line.", "Glutes tight, body like a plank."],
    "curl_range":   ["Finish with your forearm nearly vertical.", "Bring your hand up toward your shoulder.", "Squeeze at the top of the curl."],
    "curl_swing":   ["Pin your elbows to your sides.", "Stop swinging, let the biceps work.", "Lock your elbows in and stay still."],
    "too_fast":     ["Take two seconds on the way down.", "Count two seconds lowering.", "Lower slowly, count to two."],
}

DUR = {  # seconds -> (train phrasings, held-out test phrasings)
    15: (["15 seconds", "fifteen seconds"], ["quarter of a minute"]),
    30: (["30 seconds", "thirty seconds"], ["half a minute"]),
    45: (["45 seconds", "forty five seconds"], ["three quarters of a minute"]),
    60: (["60 seconds", "sixty seconds"], ["one minute", "a minute"]),
    90: (["90 seconds", "ninety seconds"], ["a minute and a half"]),
    120: (["two minutes", "120 seconds"], ["2 minutes"]),
    180: (["three minutes", "180 seconds"], ["3 minutes"]),
}
EX_PHRASES = {"squat": ["squats", "squat"], "pushup": ["push ups", "push-ups", "pushups", "push up"], "curl": ["curls", "bicep curls", "curl"]}

# (train templates, held-out test templates)
T_START = (["start {x}", "switch to {x}", "let's do {x}", "begin {x}", "i want to do {x}", "change the exercise to {x}"],
           ["can we do some {x}", "go to {x} now", "time for {x}", "move on to {x}", "okay {x} please"])
T_REST_N = (["rest {d}", "rest for {d}", "take {d} to rest", "start a {d} rest"],
            ["let me rest for {d}", "i need {d} to recover", "hold on for {d}", "pause for {d}"])
T_REST_PLAIN = (["i need a break", "take a break", "rest", "give me a rest", "time to rest"],
                ["let me catch my breath", "i need to recover a bit", "hang on i'm tired"])
T_SUMMARY = (["how was that set", "how did i do", "summary please", "give me a summary", "i'm done", "end the set", "finish the set", "how was my form"],
             ["give me my stats", "how'd that go", "wrap up the set", "tell me how i did", "what was my score"])
T_RETRIEVE = (["why do my knees hurt", "how do i fix my form", "what is good form", "how deep should i go", "why does my back hurt",
               "tips for better form", "should i go slower", "how do i stop swinging", "why are my hips sagging",
               "what should my elbows do", "how can i get deeper", "why is my chest falling forward"],
              ["is it normal for my knees to ache", "any advice on my form", "how low am i supposed to go",
               "my lower back feels off what should i do", "what's the right tempo", "why do i keep swinging my torso"])
T_EASIER = (["make it easier", "this is too hard", "go easier on me"], ["dial it back", "i'm struggling lower the difficulty"])
T_FEWER = (["fewer reps please", "reduce the reps"], ["cut down the reps", "less reps per set"])
T_MORE = (["give me more reps", "increase the reps"], ["add some reps", "let's push for more reps"])
T_LONGER = (["longer rest please", "i need more rest between sets"], ["lengthen the rest time", "give me extra rest between sets"])
T_SHORTER = (["shorter rest", "less rest between sets"], ["cut the rest time", "speed up the rest"])
T_TARGET = (["do {n} reps per set", "set the target to {n} reps"], ["let's go for {n} reps", "make it {n} reps"])
T_SETREST = (["set my rest time to {d}"], ["change the rest to {d}"])
VOICE_KINDS = ["start", "rest_n", "rest_plain", "summary", "retrieve", "easier", "fewer", "more", "longer", "shorter", "target", "setrest"]
VOICE_W = [3, 2, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1]


def js_round(x: float) -> int:
    return int(math.floor(x + 0.5))


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


# ---------------------------------------------------------------- state simulation
def simulate_mins(rng, ex: str, n: int) -> list[int]:
    """Per-rep minimum joint angle with fatigue drift (angle grows as the athlete gets shallower)."""
    _, _, base, sd, *_ = SIM[ex]
    drift = float(rng.uniform(0, 2.5)) if rng.random() < 0.6 else 0.0
    start = base + float(rng.normal(0, 6))
    return [js_round(start + drift * i + rng.normal(0, sd * 0.35)) for i in range(n)]


def sample_rep_errors(rng, ex: str, k: int) -> tuple[list[str], int]:
    """Force exactly k simultaneous errors on the last rep. Returns (ordered error ids, min angle for that rep)."""
    depth_id, form_id, _, _, depth_thr, *_ = SIM[ex]
    order = [depth_id, form_id, "too_fast"]
    chosen = set(rng.choice(3, size=k, replace=False).tolist()) if k else set()
    errs = [order[i] for i in range(3) if i in chosen]  # same order as exercises.js check()
    if depth_id in errs:
        mn = depth_thr + int(rng.integers(2, 22))
    else:
        mn = depth_thr - int(rng.integers(4, 30))
    return errs, mn


def trend(mins: list[int]) -> int:
    k = min(3, len(mins) // 2)
    if not k:
        return 0
    avg = lambda a: sum(a) / len(a)  # noqa: E731
    return js_round(avg(mins[-k:]) - avg(mins[:k]))


def make_state(rng, kb, ex, n_reps, k_errors, target, sets, set_idx, rest, with_last_cue=True) -> dict:
    mins = simulate_mins(rng, ex, n_reps)
    errs, last_min = sample_rep_errors(rng, ex, k_errors) if k_errors is not None else ([], mins[-1] if mins else 0)
    if mins and k_errors is not None:
        mins[-1] = last_min
    ex_ids = [i for i, e in kb.items() if e["ex"] in (ex, "any") and e["sev"] > 0]
    last = None
    if with_last_cue and n_reps >= 2 and rng.random() < 0.45:
        cid = pick(rng, ex_ids)
        ago = int(rng.integers(1, min(4, n_reps - 1) + 1))
        fixed = (cid not in errs) if ago == 1 else bool(rng.random() < 0.5)
        last = {"id": cid, "reps_ago": ago, "fixed": fixed}
    hist = None
    if rng.random() < 0.5:
        hist = {"sets": int(rng.integers(1, 25)), "clean_rate": round(float(rng.uniform(0.2, 0.95)), 2),
                "last": f"2026-{int(rng.integers(8, 11)):02d}-{int(rng.integers(1, 28)):02d}"}
    return {"exercise": ex, "set": set_idx, "sets": sets, "rep": n_reps, "target": target, "rest": rest,
            "errors": [{"id": i, "sev": kb[i]["sev"]} for i in errs], "last_cue": last,
            "trend": {"depth_loss_deg": trend(mins)}, "history": hist}


def knowledge_for(rng, kb, ex, wanted: list[str], noise=0.15) -> list[str]:
    """Approximate the embedding retrieval (top-2 docs for this exercise), with some retrieval noise."""
    elig = [i for i, e in kb.items() if e["ex"] in (ex, "any")]
    docs = [i for i in wanted if i in elig][:2]
    rest = [i for i in elig if i not in docs]
    while len(docs) < 2 and rest:
        docs.append(rest.pop(int(rng.integers(len(rest)))))
    if rest and rng.random() < noise:
        docs[int(rng.integers(len(docs)))] = rest[int(rng.integers(len(rest)))]
    rng.shuffle(docs)
    return docs


# ---------------------------------------------------------------- teacher policy (mirrors the SYSTEM prompt rules)
def label_rep(rng, kb, st) -> dict:
    top = sorted(st["errors"], key=lambda e: -e["sev"])[0]["id"]        # highest severity, first on ties (stable)
    lc = st["last_cue"]
    rephrase = bool(lc and lc["id"] == top and lc["fixed"] is False)
    bank = (CUE_REPHRASE if rephrase else CUE_FIRST).get(top)
    text = pick(rng, bank) if bank else " ".join((kb[top]["alt"] if rephrase else kb[top]["cue"]).split()[:12])
    return {"tool": "give_cue", "args": {"cue_id": top, "text": text}}


def label_set_end(st) -> dict:
    """Fatigue (depth lost > 12 deg) or multiple errors -> fewer reps + more rest. Otherwise keep the plan."""
    rest, target = st["rest"], st["target"]
    if st["trend"]["depth_loss_deg"] > 12 or len(st["errors"]) >= 2:
        rest, target = min(300, rest + 30), max(3, target - 2)
    return {"tool": "adjust_plan", "args": {"rest": rest, "target": target}}


def doc_for_utt(utt: str, ex: str, kb) -> list[str]:
    u = utt.lower()
    depth = {"squat": "squat_depth", "pushup": "pushup_depth", "curl": "curl_range"}[ex]
    rules = [("knee", "knee_pain"), ("swing", "curl_swing"), ("sag", "pushup_sag"), ("hips", "pushup_sag"), ("back", "squat_lean"),
             ("chest", "squat_lean"), ("forward", "squat_lean"), ("deep", depth), ("low", depth), ("slow", "too_fast"),
             ("fast", "too_fast"), ("tempo", "too_fast"), ("elbow", "pushup_depth")]
    return [d for k, d in rules if k in u][:1]


def voice_row(rng, kb, split: str, i: int) -> dict:
    s = 0 if split == "train" else 1
    ex = pick(rng, NAMES)
    target, sets = int(pick(rng, [6, 8, 10, 12])), int(pick(rng, [2, 3, 4]))
    rest = int(pick(rng, [30, 45, 60, 90, 120]))
    st = make_state(rng, kb, ex, int(rng.integers(0, target)), None, target, sets, int(rng.integers(1, sets + 1)), rest)
    st["errors"] = []
    kind = rng.choice(VOICE_KINDS, p=np.array(VOICE_W) / sum(VOICE_W))
    secs = int(pick(rng, list(DUR)))
    dphrase = pick(rng, DUR[secs][s])
    wanted: list[str] = []
    if kind == "start":
        name = pick(rng, NAMES)
        said, label = pick(rng, T_START[s]).format(x=pick(rng, EX_PHRASES[name])), {"tool": "start_exercise", "args": {"name": name}}
    elif kind == "rest_n":
        said, label = pick(rng, T_REST_N[s]).format(d=dphrase), {"tool": "rest_timer", "args": {"seconds": secs}}
    elif kind == "rest_plain":
        said, label = pick(rng, T_REST_PLAIN[s]), {"tool": "rest_timer", "args": {"seconds": rest}}
    elif kind == "summary":
        said, label = pick(rng, T_SUMMARY[s]), {"tool": "set_summary", "args": {}}
    elif kind == "retrieve":
        said = pick(rng, T_RETRIEVE[s])
        label, wanted = {"tool": "retrieve_cues", "args": {"query": said}}, doc_for_utt(said, ex, kb)
    elif kind == "easier":
        said, label = pick(rng, T_EASIER[s]), {"tool": "adjust_plan", "args": {"rest": min(300, rest + 30), "target": max(3, target - 2)}}
    elif kind == "fewer":
        said, label = pick(rng, T_FEWER[s]), {"tool": "adjust_plan", "args": {"rest": rest, "target": max(3, target - 2)}}
    elif kind == "more":
        said, label = pick(rng, T_MORE[s]), {"tool": "adjust_plan", "args": {"rest": rest, "target": min(20, target + 2)}}
    elif kind == "longer":
        said, label = pick(rng, T_LONGER[s]), {"tool": "adjust_plan", "args": {"rest": min(300, rest + 30), "target": target}}
    elif kind == "shorter":
        said, label = pick(rng, T_SHORTER[s]), {"tool": "adjust_plan", "args": {"rest": max(10, rest - 15), "target": target}}
    elif kind == "target":
        n = int(rng.integers(5, 16))
        said, label = pick(rng, T_TARGET[s]).format(n=n), {"tool": "adjust_plan", "args": {"rest": rest, "target": n}}
    else:  # setrest
        said, label = pick(rng, T_SETREST[s]).format(d=dphrase), {"tool": "adjust_plan", "args": {"rest": secs, "target": target}}
    return {"id": f"{split}-voice-{i}", "slice": "voice", "kind": "VOICE", "allow": ALLOW_VOICE, "said": said, "state": st,
            "knowledge": knowledge_for(rng, kb, ex, wanted), "label": label, "source": "synthetic"}


def rep_row(rng, kb, split: str, i: int, set_end: bool) -> dict:
    ex = pick(rng, NAMES)
    target, sets = int(pick(rng, [6, 8, 10, 12])), int(pick(rng, [2, 3, 4]))
    rest = int(pick(rng, [30, 45, 60, 90, 120]))
    n = target if set_end else int(rng.integers(1, target))
    k = int(rng.choice([0, 1, 2, 3], p=[.35, .4, .2, .05])) if set_end else int(rng.choice([1, 2, 3], p=[.6, .3, .1]))
    st = make_state(rng, kb, ex, n, k, target, sets, int(rng.integers(1, sets + 1)), rest)
    wanted = [e["id"] for e in st["errors"]] or [ex]
    label = label_set_end(st) if set_end else label_rep(rng, kb, st)
    return {"id": f"{split}-{'set' if set_end else 'rep'}-{i}", "slice": "set_end" if set_end else "rep_cue", "kind": "REP",
            "allow": ALLOW_SET_END if set_end else ALLOW_REP, "said": "", "state": st,
            "knowledge": knowledge_for(rng, kb, ex, wanted), "label": label, "source": "synthetic"}


def rows_from_logs(rng, kb, paths: list[str]) -> list[dict]:
    """Real states from 'Download session log' JSON files, relabelled by the teacher policy."""
    out = []
    for p in paths:
        for e in json.load(open(p, encoding="utf-8")):
            st = e.get("state")
            if not st or any(x["id"] not in kb for x in st["errors"]):
                continue
            set_end = st["rep"] >= st["target"]
            if not set_end and not st["errors"]:
                continue
            ex = st["exercise"]
            wanted = [x["id"] for x in st["errors"]] or [ex]
            out.append({"id": f"log-{len(out)}", "slice": "set_end" if set_end else "rep_cue", "kind": "REP", "said": "",
                        "allow": ALLOW_SET_END if set_end else ALLOW_REP, "state": st,
                        "knowledge": knowledge_for(rng, kb, ex, wanted),
                        "label": label_set_end(st) if set_end else label_rep(rng, kb, st), "source": "log"})
    return out


def build_split(rng, kb, split: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        r = rng.random()
        rows.append(rep_row(rng, kb, split, i, False) if r < 0.45 else rep_row(rng, kb, split, i, True) if r < 0.70 else voice_row(rng, kb, split, i))
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(js_dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--logs", nargs="*", default=[], help="session-log JSON files (glob ok)")
    ap.add_argument("--refresh-test", action="store_true", help="overwrite the frozen test set (old metrics become invalid)")
    a = ap.parse_args()
    kb = load_kb()
    DATA_DIR.mkdir(exist_ok=True)
    train = build_split(np.random.default_rng([a.seed, 0]), kb, "train", a.n_train)
    logs = [p for g in a.logs for p in glob.glob(g)]
    if logs:
        extra = rows_from_logs(np.random.default_rng([a.seed, 9]), kb, logs)
        print(f"mixed in {len(extra)} real-log rows from {len(logs)} file(s)")
        train += extra
    np.random.default_rng([a.seed, 3]).shuffle(train)
    write(DATA_DIR / "train.jsonl", train)
    write(DATA_DIR / "val.jsonl", build_split(np.random.default_rng([a.seed, 1]), kb, "val", a.n_val))
    test = DATA_DIR / "test.jsonl"
    if a.refresh_test or not test.exists():
        write(test, build_split(np.random.default_rng([a.seed, 2]), kb, "test", a.n_test))
        print("wrote NEW frozen test set")
    else:
        print("kept existing frozen test set")
    print(f"train={len(train)} val={a.n_val} test sha={file_sha(test)}")


if __name__ == "__main__":
    main()
