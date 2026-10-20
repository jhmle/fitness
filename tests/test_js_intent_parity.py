"""eval.py's rules baseline must answer exactly like agent.js regexIntent, or the baseline numbers describe a different system."""
import itertools
import json
import pathlib
import shutil
import subprocess

import pytest

import gen_data as g
from common import AGENT_JS, DATA_DIR, load_rows
from eval import predict_rules

EXTRA = ["make it 4 reps", "do twelve reps per set", "rest for 2 minutes", "set my rest time to a minute and a half", "take a break", "I need more rest",
         "shorter rest please", "reduce the reps", "give me more reps", "this is too hard", "how was that set", "why do my knees hurt", "banana",
         "change the exercise to squats", "i want to do curls", "start a 30 second rest", "rest for half a minute", "let me rest for thirty seconds"]


def _utterances():
    rows = [r for r in load_rows(DATA_DIR / "test.jsonl") if r["slice"] == "voice"]
    items = [{"said": r["said"], "rest": r["state"]["rest"], "target": r["state"]["target"]} for r in rows]
    for name in ("T_START", "T_REST_PLAIN", "T_SUMMARY", "T_RETRIEVE", "T_EASIER", "T_FEWER", "T_MORE", "T_LONGER", "T_SHORTER"):
        for tpl in itertools.chain(*getattr(g, name)):
            items.append({"said": tpl.format(x="squats") if "{x}" in tpl else tpl, "rest": 60, "target": 8})
    for tpl in itertools.chain(*g.T_TARGET):
        items.append({"said": tpl.format(n=10), "rest": 60, "target": 8})
    for tpl, d in itertools.product(itertools.chain(*g.T_REST_N, *g.T_SETREST), ["30 seconds", "two minutes", "a minute and a half", "quarter of a minute"]):
        items.append({"said": tpl.format(d=d), "rest": 60, "target": 8})
    items += [{"said": s, "rest": 60, "target": 8} for s in EXTRA] + [{"said": s, "rest": 290, "target": 19} for s in EXTRA]   # also exercise the clamps
    return items


def _norm(c):
    return None if c is None else {"tool": c["tool"], "args": c.get("args") or {}}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_python_rules_match_agent_js_regex_intent(tmp_path):
    items = _utterances()
    shutil.copy(AGENT_JS, tmp_path / "agent.mjs")
    (tmp_path / "items.json").write_text(json.dumps(items))
    script = pathlib.Path(__file__).parent / "js_intent_dump.mjs"
    out = subprocess.run(["node", str(script), str(tmp_path / "agent.mjs"), str(tmp_path / "items.json"), str(tmp_path / "out.json")],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    js = json.loads((tmp_path / "out.json").read_text())
    for it, j in zip(items, js):
        row = {"kind": "VOICE", "said": it["said"], "state": {"rest": it["rest"], "target": it["target"]}}
        assert _norm(predict_rules(row)) == _norm(j), it
