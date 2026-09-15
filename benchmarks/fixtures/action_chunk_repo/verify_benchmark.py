import ast
import json
import re
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


def _fixture_fact(fixture_root, name):
    def read(path):
        return (fixture_root / path).read_text(encoding="utf-8")

    if name == "default_timeout":
        match = re.search(r"^DEFAULT_TIMEOUT\s*=\s*(\d+)", read(Path("app/config.py")), re.MULTILINE)
        assert match, "DEFAULT_TIMEOUT was not found in the fixture"
        return {"values": [match.group(1)]}
    if name == "retry_limit":
        match = re.search(r"^RETRY_LIMIT\s*=\s*(\d+)", read(Path("app/config.py")), re.MULTILINE)
        assert match, "RETRY_LIMIT was not found in the fixture"
        return {"values": [match.group(1)]}
    if name == "parser_normalization":
        source = read(Path("app/parser.py"))
        test = read(Path("tests/test_parser.py"))
        assert ").upper()" in source
        match = re.search(r"method\s*==\s*[\"']([A-Z]+)[\"']", test)
        assert match, "parser normalization result was not found in the fixture"
        return {"values": [match.group(1)]}
    if name == "parse_result_shape":
        source = read(Path("app/parser.py"))
        fields = [field for field in ("method", "route", "body") if f"self.{field}" in source]
        assert fields == ["method", "route", "body"]
        return {"values": fields}
    if name == "response_shape":
        source = read(Path("app/formatter.py"))
        match = re.search(r"[\"']status[\"']\s*:\s*(\d+)", source)
        assert match, "response status was not found in the fixture"
        assert "dict(payload)" in source
        return {"values": [match.group(1), "data"]}
    if name == "validation_precondition":
        source = read(Path("app/validator.py"))
        messages = re.findall(r"raise ValueError\([\"']([^\"']+)[\"']\)", source)
        match = next((message for message in messages if "route" in message), None)
        assert match, "validation error was not found in the fixture"
        return {"values": [match]}
    if name == "request_order":
        source = read(Path("app/service.py"))
        tree = ast.parse(source, filename="app/service.py")
        handle_request = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "handle_request"
            ),
            None,
        )
        assert handle_request is not None, "handle_request was not found in the fixture"

        expected = ("validate_request", "route_request", "get_or_put", "format_response")
        order = []

        class _RequestOrderVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                function = node.func
                called_name = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else None
                )
                if called_name in expected:
                    order.append(called_name)
                self.generic_visit(node)

            def visit_Lambda(self, node):
                return None

            def visit_FunctionDef(self, node):
                return None

            def visit_AsyncFunctionDef(self, node):
                return None

            def visit_ClassDef(self, node):
                return None

        visitor = _RequestOrderVisitor()
        for statement in handle_request.body:
            visitor.visit(statement)
        assert order == ["validate_request", "route_request", "get_or_put", "format_response"]
        return {"values": order, "ordered": order}
    if name == "route_health":
        source = read(Path("app/router.py"))
        match = re.search(r"[\"']GET /health[\"']\s*:\s*[\"']([^\"']+)[\"']", source)
        assert match, "health route was not found in the fixture"
        return {"values": [match.group(1)]}
    if name == "source_metadata":
        source = read(Path("app/loader.py"))
        assert 'record["source"] = path' in source
        return {"values": ["source", "path"]}
    if name == "cache_reuse":
        source = read(Path("app/cache.py"))
        assert all(f"def {method}" in source for method in ("get", "put", "get_or_put"))
        return {"values": ["get", "put", "get_or_put"]}
    if name == "catalog_marker":
        source = read(Path("app/catalog.py"))
        match = re.search(r"CATALOG_HEADER\s*=\s*[\"']([^\"']+)[\"']", source)
        assert match, "catalog marker was not found in the fixture"
        return {"values": [match.group(1)]}
    raise AssertionError(f"unknown semantic fixture fact: {name}")


def check_semantic(tokens=(), minimum_actions=0, paths=(), patterns=(), ordered_paths=(), facts=()):
    """Check tool evidence and facts derived from the inspected fixture."""
    check(tokens, minimum_actions, paths=paths, patterns=patterns, ordered_paths=ordered_paths)
    report, _ = _latest_run()
    answer = str(report["final_answer"]).lower()
    fixture_root = Path.cwd()
    for name in facts:
        fact = _fixture_fact(fixture_root, str(name))
        for value in fact.get("values", []):
            assert str(value).lower() in answer, (name, value)
        ordered = fact.get("ordered", [])
        if ordered:
            positions = [answer.index(str(value).lower()) for value in ordered]
            assert positions == sorted(positions), (name, ordered)
