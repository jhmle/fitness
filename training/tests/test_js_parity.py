"""The training prompt must be byte-identical to what the browser sends. Runs the real agent.js under Node."""
import json
import pathlib
import shutil
import subprocess

import pytest

from common import AGENT_JS, DATA_DIR, load_kb, load_rows, load_system, to_messages


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_prompts_match_agent_js(tmp_path):
    rows = load_rows(DATA_DIR / "test.jsonl")
    pick = [r for s in ("rep_cue", "set_end", "voice") for r in [x for x in rows if x["slice"] == s][:25]]
    shutil.copy(AGENT_JS, tmp_path / "agent.mjs")
    (tmp_path / "rows.json").write_text(json.dumps(pick))
    script = pathlib.Path(__file__).parent / "js_prompt_dump.mjs"
    out = subprocess.run(["node", str(script), str(tmp_path / "agent.mjs"), str(tmp_path / "rows.json"), str(tmp_path / "out.json")],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    js = json.loads((tmp_path / "out.json").read_text())
    kb, system = load_kb(), load_system()
    for r, msgs in zip(pick, js):
        assert msgs == to_messages(r, kb, system), r["id"]
