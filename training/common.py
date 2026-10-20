"""Shared helpers for data generation, training and evaluation.

Single source of truth: the SYSTEM prompt and the knowledge base (KB) are *parsed from agent.js*,
so the model is always trained on exactly the prompt the browser will send. If you edit agent.js,
regenerate the data (gen_data.py) and the tests will tell you if anything drifted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _find(name):
    """App files may live in web/, docs/ or the repo root (GitHub Pages layouts differ)."""
    for p in (ROOT / "web" / name, ROOT / name, ROOT / "docs" / name):
        if p.exists():
            return p
    return ROOT / "web" / name


AGENT_JS = pathlib.Path(os.getenv("AGENT_JS") or _find("agent.js"))
APP_DIR = AGENT_JS.parent
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
METRICS_DIR = pathlib.Path(__file__).resolve().parent / "metrics"
NAMES = ["squat", "pushup", "curl"]

# Tool whitelists, mirroring agent.js onRep()/onVoice().
ALLOW_REP = ["give_cue"]
ALLOW_SET_END = ["adjust_plan"]
ALLOW_VOICE = ["retrieve_cues", "adjust_plan", "rest_timer", "start_exercise", "set_summary"]


def _src() -> str:
    if not AGENT_JS.exists():
        raise FileNotFoundError(f"{AGENT_JS} not found. Looked in web/, docs/ and the repo root. Set AGENT_JS=/path/to/agent.js")
    return AGENT_JS.read_text(encoding="utf-8")


def load_system() -> str:
    m = re.search(r"const SYSTEM = `(.*?)`;", _src(), re.S)
    if not m:
        raise ValueError("could not find `const SYSTEM = ...` in agent.js")
    return m.group(1)


def load_kb() -> dict:
    pat = re.compile(
        r'(\w+):\s*\{\s*ex:\s*"([^"]*)",\s*sev:\s*(\d+),\s*cue:\s*"([^"]*)",\s*alt:\s*"([^"]*)",\s*why:\s*"([^"]*)"'
    )
    kb = {m[1]: {"ex": m[2], "sev": int(m[3]), "cue": m[4], "alt": m[5], "why": m[6]} for m in pat.finditer(_src())}
    if not kb:
        raise ValueError("could not parse KB from agent.js")
    return kb


# ---------------------------------------------------------------- prompt building (must match agent.js ask())
def _norm(o):
    """Make Python JSON match JS JSON.stringify: integral floats print as ints, 2-dp floats."""
    if isinstance(o, float):
        o = round(o, 2)
        return int(o) if o == int(o) else o
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    return o


def js_dumps(o) -> str:
    return json.dumps(_norm(o), separators=(",", ":"), ensure_ascii=False)


def build_user(kind: str, state: dict, docs: list[dict], said: str = "") -> str:
    said_part = f' SAID "{said}"' if said else ""
    know = " | ".join(f"{d['id']}: {d['cue']} {d['alt']}" for d in docs)
    return f"EVENT {kind}{said_part}\nSTATE {js_dumps(state)}\nKNOWLEDGE {know}"


def to_messages(row: dict, kb: dict | None = None, system: str | None = None) -> list[dict]:
    kb = kb or load_kb()
    docs = [{"id": i, **kb[i]} for i in row["knowledge"]]
    return [
        {"role": "system", "content": system if system is not None else load_system()},
        {"role": "user", "content": build_user(row["kind"], row["state"], docs, row.get("said", ""))},
    ]


def target_text(label: dict) -> str:
    return json.dumps(label, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------- validate() port (agent.js validate)
_MISSING = object()


def _num(v) -> float:
    """Approximates JS unary plus: +undefined=NaN, +null=0, +"60"=60."""
    if v is _MISSING:
        return math.nan
    if v is None:
        return 0.0
    if isinstance(v, (bool, int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return 0.0
        try:
            return float(s)
        except ValueError:
            return math.nan
    return math.nan


def _in(v, lo, hi) -> bool:
    x = _num(v)
    return math.isfinite(x) and lo <= x <= hi


def _int_if_whole(x: float):
    return int(x) if x == int(x) else x


def validate(c, allow: list[str], errors: list[str]):
    """Never trust model output: whitelist the tool and check every arg. Returns a clean call or None."""
    if not isinstance(c, dict) or not isinstance(c.get("tool"), str) or c["tool"] not in allow:
        return None
    a = c.get("args") or {}
    if not isinstance(a, dict):
        a = {}
    g = lambda k: a.get(k, _MISSING)  # noqa: E731
    t = c["tool"]
    if t == "give_cue":
        if g("cue_id") not in errors:
            return None
        text = g("text")
        text = re.sub(r"[^\w ,.'!?-]", "", text, flags=re.A).strip()[:100] if isinstance(text, str) else ""
        return {"tool": t, "args": {"cue_id": a["cue_id"], "text": text}}
    if t == "adjust_plan":
        if _in(g("rest"), 10, 300) and _in(g("target"), 3, 20):
            return {"tool": t, "args": {"rest": _int_if_whole(_num(g("rest"))), "target": _int_if_whole(_num(g("target")))}}
        return None
    if t == "retrieve_cues":
        q = g("query")
        return c if isinstance(q, str) and len(q) < 200 else None
    if t == "rest_timer":
        return {"tool": t, "args": {"seconds": _int_if_whole(_num(g("seconds")))}} if _in(g("seconds"), 10, 300) else None
    if t == "start_exercise":
        return c if g("name") in NAMES else None
    if t == "set_summary":
        return {"tool": t, "args": {}}
    return None


def extract_call(text) -> dict | None:
    """Same parsing as agent.js: first '{' to last '}' then JSON.parse."""
    m = re.search(r"\{[\s\S]*\}", str(text or ""))
    if not m:
        return None
    try:
        c = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return c if isinstance(c, dict) else None


# ---------------------------------------------------------------- data io
def load_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def row_errors(row: dict) -> list[str]:
    return [e["id"] for e in row["state"]["errors"]]
