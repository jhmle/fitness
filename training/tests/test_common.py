from common import NAMES, extract_call, js_dumps, load_kb, load_system, validate


def test_system_and_kb_parse_from_agent_js():
    s = load_system()
    assert "decision module of a workout coach" in s and '"tool":"give_cue"' in s
    kb = load_kb()
    assert {"squat_depth", "pushup_sag", "curl_swing", "too_fast", "knee_pain"} <= set(kb)
    assert kb["squat_lean"]["sev"] == 3 and kb["too_fast"]["ex"] == "any"


def test_js_dumps_matches_json_stringify_style():
    assert js_dumps({"a": 1.0, "b": 0.5, "c": [True, None], "d": 0.78}) == '{"a":1,"b":0.5,"c":[true,null],"d":0.78}'


def test_validate_give_cue():
    # characters outside [\w ,.'!?-] are stripped one by one, exactly like the JS regex
    ok = validate({"tool": "give_cue", "args": {"cue_id": "squat_lean", "text": "Chest <b>up</b>!"}}, ["give_cue"], ["squat_lean"])
    assert ok == {"tool": "give_cue", "args": {"cue_id": "squat_lean", "text": "Chest bupb!"}}
    assert validate({"tool": "give_cue", "args": {"cue_id": "squat_depth"}}, ["give_cue"], ["squat_lean"]) is None   # not an active error
    assert validate({"tool": "give_cue", "args": {"cue_id": "squat_lean"}}, ["adjust_plan"], ["squat_lean"]) is None  # tool not allowed


def test_validate_numeric_ranges():
    allow = ["adjust_plan", "rest_timer"]
    assert validate({"tool": "adjust_plan", "args": {"rest": 90, "target": 6}}, allow, []) is not None
    assert validate({"tool": "adjust_plan", "args": {"rest": 5, "target": 6}}, allow, []) is None
    assert validate({"tool": "adjust_plan", "args": {"rest": 90, "target": 25}}, allow, []) is None
    assert validate({"tool": "adjust_plan", "args": {"rest": "90", "target": "6"}}, allow, [])["args"] == {"rest": 90, "target": 6}
    assert validate({"tool": "rest_timer", "args": {"seconds": 301}}, allow, []) is None
    assert validate({"tool": "rest_timer", "args": {}}, allow, []) is None


def test_validate_exercise_and_query():
    allow = ["start_exercise", "retrieve_cues", "set_summary"]
    assert validate({"tool": "start_exercise", "args": {"name": "pushup"}}, allow, [])
    assert validate({"tool": "start_exercise", "args": {"name": "deadlift"}}, allow, []) is None
    assert validate({"tool": "retrieve_cues", "args": {"query": "x" * 200}}, allow, []) is None
    assert validate({"tool": "set_summary"}, allow, []) == {"tool": "set_summary", "args": {}}
    assert set(NAMES) == {"squat", "pushup", "curl"}


def test_extract_call_like_the_browser():
    assert extract_call('Sure! {"tool":"set_summary","args":{}} done')["tool"] == "set_summary"
    assert extract_call("no json here") is None
    assert extract_call('{"tool": broken}') is None
