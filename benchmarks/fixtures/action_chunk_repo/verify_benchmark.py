import json
from pathlib import Path


def _latest_run():
    reports = sorted(Path(".pico/runs").glob("*/report.json"), key=lambda p: str(p))
    assert reports, "missing Pico report"
    report_path = reports[-1]
    trace_path = report_path.with_name("trace.jsonl")
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    return json.loads(report_path.read_text(encoding="utf-8")), events


def check(tokens, minimum_actions, paths=(), patterns=(), ordered_paths=()):
    report, events = _latest_run()
    assert report["status"] == "completed"
    assert report["stop_reason"] == "final_answer_returned"
    state = report["task_state"]
    primitive_submissions = state.get(
        "primitive_submissions",
        state.get("primitive_tool_calls", state.get("tool_steps", 0)),
    )
    assert primitive_submissions >= minimum_actions
    actions = [event for event in events if event.get("event") == "tool_executed"]
    observed_paths = {event.get("args", {}).get("path") for event in actions}
    for path in paths:
        assert path in observed_paths, (path, sorted(observed_paths))
    observed_patterns = {event.get("args", {}).get("pattern") for event in actions}
    for pattern in patterns:
        assert pattern in observed_patterns, (pattern, sorted(observed_patterns))
    if ordered_paths:
        action_paths = [event.get("args", {}).get("path") for event in actions]
        positions = [action_paths.index(path) for path in ordered_paths]
        assert positions == sorted(positions), (ordered_paths, action_paths)
    answer = str(report["final_answer"]).lower()
    for token in tokens:
        assert str(token).lower() in answer, token
