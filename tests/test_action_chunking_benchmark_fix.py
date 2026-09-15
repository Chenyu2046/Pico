import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pico.evaluation.evaluator import aggregate_provider_usage
from scripts import run_action_chunking_real_benchmark as runner


def _write_artifact(path, rows):
    path = Path(path)
    group = path.stem
    task_ids = [row["id"] for row in rows]
    prompt_snapshot_id = "fix-pilot-prompt"
    payload = {
        "artifact_id": f"{group}-rep-01",
        "group": group,
        "repetition": 1,
        "runtime": {"commit_sha": "fix-pilot-commit"},
        "reproducibility": {
            "model_name": "gpt-5.6-luna",
            "model_version": "openai-compatible-responses",
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 768},
            "task_ids": task_ids,
            "task_prompt_snapshot_id": prompt_snapshot_id,
            "prompt_snapshot_id": prompt_snapshot_id,
            "fixture_snapshot_id": "fix-pilot-fixture",
            "step_budget_summary": {
                "min": 8,
                "max": 8,
                "unique": [8],
                "count": len(task_ids),
            },
            "execution_config": {
                "continue_on_error": False,
                "allowed_tools_by_task": {
                    task_id: ["list_files", "read_file", "search"] for task_id in task_ids
                },
            },
            "action_chunking": runner.normalize_action_chunking(runner.GROUP_CONFIGS[group]),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _expected_contract(artifact_paths):
    reproducibility = json.loads(
        Path(artifact_paths["A"][0]).read_text(encoding="utf-8")
    )["reproducibility"]
    return {
        field: reproducibility[field]
        for field in ("task_prompt_snapshot_id", "fixture_snapshot_id", "step_budget_summary")
    }


def _pilot_row(task_id, group, index, *, chunked=True):
    positive = {"id": task_id, "category": "known_path_multi_file_inspection"}
    return {
        **positive,
        "status": "pass",
        "passed": True,
        "verifier_passed": True,
        "failure_category": None,
        "logical_decisions": 10 if group == "A" else 8,
        "provider_requests": 10 if group == "A" else 8,
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "cached_tokens": 20,
        "token_usage_coverage": 1.0,
        "provider_attempts": [
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "cached_tokens": 20,
            }
        ],
        "chunk_count": int(group != "A" and chunked),
        "chunk_lengths": [2] if group != "A" and chunked else [],
        "chunk_interrupt_rate": 0.0,
        "e2e_latency_ms": 10,
    }


def _run_verifier_in_fixture(fixture, command):
    return subprocess.run(
        [sys.executable, "-c", command],
        cwd=fixture,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_report(fixture, final_answer, tool_paths=()):
    run_dir = Path(fixture) / ".pico" / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stop_reason": "final_answer_returned",
                "final_answer": final_answer,
                "task_state": {"primitive_submissions": len(tool_paths)},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trace.jsonl").write_text(
        "".join(
            json.dumps(
                {"event": "tool_executed", "args": {"path": path}}
            )
            + "\n"
            for path in tool_paths
        ),
        encoding="utf-8",
    )


def test_primary_environment_records_one_current_commit_for_a_b_c(tmp_path):
    _, summary = runner.run_real_benchmark(
        benchmark_path="benchmarks/action_chunking_tasks.json",
        groups=["A", "B", "C"],
        repetitions=1,
        temperature=0.0,
        max_new_tokens=768,
        timeout=1,
        output_dir=tmp_path,
        api_key="",
        base_url="https://api.example.invalid/v1",
        task_ids=["MF-01"],
    )

    environment = json.loads(next(tmp_path.glob("*/environment.json")).read_text(encoding="utf-8"))
    current_head = environment["commit_sha"]
    assert environment["group_commits"] == {"A": current_head, "B": current_head, "C": current_head}
    assert summary["primary_ablation"]["same_commit"] is False


def test_provider_usage_aggregation_sums_complete_attempts_and_preserves_missing_values():
    attempts = [
        {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cached_tokens": 20},
        {"input_tokens": 120, "output_tokens": 20, "total_tokens": 140, "cached_tokens": 30},
    ]
    assert aggregate_provider_usage(attempts) == {
        "input_tokens": 220,
        "output_tokens": 30,
        "total_tokens": 250,
        "cached_tokens": 50,
        "token_usage_coverage": 1.0,
        "token_usage_complete": True,
    }

    incomplete = aggregate_provider_usage([attempts[0], {"input_tokens": None, "output_tokens": 20, "total_tokens": 140}])
    assert incomplete["input_tokens"] is None
    assert incomplete["output_tokens"] is None
    assert incomplete["total_tokens"] is None
    assert incomplete["token_usage_coverage"] == 0.5
    assert incomplete["token_usage_complete"] is False


def test_semantic_verifier_rejects_prompt_echo_without_file_evidence(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    import shutil

    fixture.mkdir()
    shutil.copy("benchmarks/fixtures/action_chunk_repo/verify_benchmark.py", fixture / "verify_benchmark.py")
    monkeypatch.chdir(fixture)
    _write_report(fixture, "The actual timeout is 30.")
    command = (
        "from verify_benchmark import check_semantic; "
        "check_semantic(['DEFAULT_TIMEOUT'], 1, paths=['app/config.py'], facts=['default_timeout'])"
    )
    result = _run_verifier_in_fixture(fixture, command)
    assert result.returncode != 0


def test_semantic_verifier_requires_facts_derived_from_fixture(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    source = Path("benchmarks/fixtures/action_chunk_repo")
    import shutil

    shutil.copytree(source, fixture)
    monkeypatch.chdir(fixture)
    _write_report(
        fixture,
        "The parser normalizes to GET; the response has status 200 and data. "
        "The request order is validate_request -> route_request -> get_or_put -> format_response.",
        ["app/parser.py", "app/formatter.py", "app/service.py"],
    )
    command = (
        "from verify_benchmark import check_semantic; "
        "check_semantic([], 3, paths=['app/parser.py', 'app/formatter.py', 'app/service.py'], "
        "facts=['parser_normalization', 'response_shape', 'request_order'])"
    )
    result = _run_verifier_in_fixture(fixture, command)
    assert result.returncode == 0, result.stderr


def test_pico_tool_and_chunk_smoke_exercise_real_protocol_shapes():
    tool_result = runner._run_pico_smoke(
        runner.FakeModelClient(
            [
                '<tool>{"name":"read_file","args":{"path":"app/config.py","start":1,"end":20}}</tool>',
                "<final>The timeout is 30.</final>",
            ]
        ),
        action_chunking={"enabled": False},
        allowed_tools=["read_file"],
        prompt="Read app/config.py and report the actual timeout value.",
    )
    assert tool_result["status"] == "passed"
    assert tool_result["primitive_submissions"] >= 1
    assert tool_result["stop_reason"] == "final_answer_returned"

    chunk_result = runner._run_pico_smoke(
        runner.FakeModelClient(
            [
                runner._chunk(
                    [
                        runner._tool("read_file", {"path": "app/config.py", "start": 1, "end": 20}),
                        runner._tool("read_file", {"path": "app/parser.py", "start": 1, "end": 20}),
                    ]
                ),
                "<final>The files were inspected.</final>",
            ]
        ),
        action_chunking=runner.GROUP_CONFIGS["B"],
        allowed_tools=runner.ALLOWED_TOOLS,
        prompt="Inspect app/config.py and app/parser.py; they are independent.",
    )
    assert chunk_result["status"] == "passed"
    assert chunk_result["chunk_count"] > 0
    assert chunk_result["mean_chunk_length"] >= 2


def test_preflight_requires_provider_tool_and_chunk_gates():
    clients = iter(
        [
            runner._DeterministicModelClient(["PICO_RESPONSES_PREFLIGHT_OK"]),
            runner.FakeModelClient(
                [
                    '<tool>{"name":"read_file","args":{"path":"app/config.py","start":1,"end":20}}</tool>',
                    "<final>The timeout is 30.</final>",
                ]
            ),
            runner.FakeModelClient(
                [
                    runner._chunk(
                        [
                            runner._tool("read_file", {"path": "app/config.py", "start": 1, "end": 20}),
                            runner._tool("read_file", {"path": "app/parser.py", "start": 1, "end": 20}),
                        ]
                    ),
                    "<final>Inspected.</final>",
                ]
            ),
        ]
    )
    result = runner._preflight("local-test-key", "https://api.example.invalid/v1", 1, client_factory=lambda: next(clients))
    assert result["status"] == "passed"
    assert result["provider"]["token_usage_complete"] is True
    assert result["pico_tool_smoke"]["primitive_submissions"] >= 1
    assert result["pico_chunk_smoke"]["chunk_count"] > 0


def test_preflight_keeps_successful_308_diagnostic_without_breaking_usage_gate():
    class _CanonicalizedProviderClient(runner._DeterministicModelClient):
        def complete(self, prompt, max_new_tokens, **kwargs):
            response = super().complete(prompt, max_new_tokens, **kwargs)
            self.last_provider_metadata["provider_requests"] = 2
            self.last_redirect_diagnostic = {
                "request_url": "https://api.example.invalid/v1/responses",
                "http_status": 308,
                "location": "https://api.example.invalid/v1/responses/",
                "location_same_host": True,
                "response_body_prefix": "canonical endpoint",
            }
            return response

    clients = iter(
        [
            _CanonicalizedProviderClient(["PICO_RESPONSES_PREFLIGHT_OK"]),
            runner.FakeModelClient(
                [
                    '<tool>{"name":"read_file","args":{"path":"app/config.py","start":1,"end":20}}</tool>',
                    "<final>The timeout is 30.</final>",
                ]
            ),
            runner.FakeModelClient(
                [
                    runner._chunk(
                        [
                            runner._tool("read_file", {"path": "app/config.py", "start": 1, "end": 20}),
                            runner._tool("read_file", {"path": "app/parser.py", "start": 1, "end": 20}),
                        ]
                    ),
                    "<final>Inspected.</final>",
                ]
            ),
        ]
    )
    result = runner._preflight(
        "local-test-key",
        "https://api.example.invalid/v1",
        1,
        client_factory=lambda: next(clients),
    )

    assert result["status"] == "passed"
    assert result["provider"]["provider_requests"] == 2
    assert result["provider"]["token_usage_complete"] is True
    assert result["provider"]["http_308"]["location_same_host"] is True


class _TimeoutSmokeClient(runner.FakeModelClient):
    def complete(self, prompt, max_new_tokens, **kwargs):
        self.last_provider_attempts = [{"status": "error", "error": "The read operation timed out"}]
        self.last_completion_metadata = {}
        raise TimeoutError("The read operation timed out")


class _TimeoutAfterToolSmokeClient(runner.FakeModelClient):
    def complete(self, prompt, max_new_tokens, **kwargs):
        if self.outputs:
            return super().complete(prompt, max_new_tokens, **kwargs)
        self.last_provider_attempts = [{"status": "error", "error": "The read operation timed out"}]
        self.last_completion_metadata = {}
        raise TimeoutError("The read operation timed out")


def test_smoke_diagnostics_distinguish_provider_and_tool_failures():
    initial_timeout = runner._run_pico_smoke(
        _TimeoutSmokeClient([]),
        action_chunking={"enabled": False},
        allowed_tools=["read_file"],
        prompt="Read app/config.py.",
    )
    assert initial_timeout["diagnostics"]["failure_class"] == "initial_provider_request_timeout"
    assert initial_timeout["diagnostics"]["tool_observation_count"] == 0

    after_observation_timeout = runner._run_pico_smoke(
        _TimeoutAfterToolSmokeClient(
            ['<tool>{"name":"read_file","args":{"path":"app/config.py","start":1,"end":20}}</tool>']
        ),
        action_chunking={"enabled": False},
        allowed_tools=["read_file"],
        prompt="Read app/config.py.",
    )
    assert after_observation_timeout["diagnostics"]["failure_class"] == "provider_request_timeout_after_tool_observation"
    assert after_observation_timeout["diagnostics"]["tool_observation_count"] == 1

    tool_failure = runner._run_pico_smoke(
        runner.FakeModelClient(
            [
                '<tool>{"name":"read_file","args":{"path":"missing.py","start":1,"end":20}}</tool>',
                "<final>Done.</final>",
            ]
        ),
        action_chunking={"enabled": False},
        allowed_tools=["read_file"],
        prompt="Read missing.py.",
    )
    assert tool_failure["status"] == "failed"
    assert tool_failure["diagnostics"]["failure_class"] == "tool_execution_failure"
    assert tool_failure["diagnostics"]["tool_failure_count"] == 1


def test_pilot_gate_passes_without_promoting_to_final_conclusion(tmp_path):
    task_ids = [f"T-{index}" for index in range(10)]
    artifact_paths = {}
    for group in ("A", "B", "C"):
        path = tmp_path / f"{group}.json"
        _write_artifact(path, [_pilot_row(task_id, group, index) for index, task_id in enumerate(task_ids)])
        artifact_paths[group] = [path]

    summary = runner.summarize_real_artifacts(
        artifact_paths,
        task_ids,
        mode="pilot",
        expected_contract=_expected_contract(artifact_paths),
    )
    assert summary["pilot_status"] == "PASS"
    assert summary["final_conclusion"] == "INCONCLUSIVE"

    _write_artifact(
        artifact_paths["B"][0],
        [_pilot_row(task_id, "B", index, chunked=index == 0) for index, task_id in enumerate(task_ids)],
    )
    failed = runner.summarize_real_artifacts(
        artifact_paths,
        task_ids,
        mode="pilot",
        expected_contract=_expected_contract(artifact_paths),
    )
    assert failed["pilot_status"] == "FAIL"
    assert failed["pilot_reason"] == "insufficient_chunk_adoption"


def test_invalid_token_coverage_disables_token_reduction(tmp_path):
    task_ids = [f"T-{index}" for index in range(20)]
    artifact_paths = {}
    for group in ("A", "B", "C"):
        path = tmp_path / f"{group}.json"
        row = _pilot_row(task_ids[0], group, 0)
        if group == "A":
            row["provider_attempts"] = [{"input_tokens": None, "output_tokens": 10, "total_tokens": 110}]
            row["input_tokens"] = None
            row["total_tokens"] = None
            row["token_usage_coverage"] = 0.0
        _write_artifact(path, [row])
        artifact_paths[group] = [path]

    summary = runner.summarize_real_artifacts(
        artifact_paths,
        task_ids,
        expected_contract=_expected_contract(artifact_paths),
    )
    assert summary["gates"]["token_metric_valid"] is False
    assert summary["comparisons"]["B_vs_A"]["input_tokens"]["reduction_pct"] is None


def test_historical_worktree_context_cleans_up_even_when_run_fails(tmp_path, monkeypatch):
    worktree = tmp_path / "historical-worktree"
    calls = []

    def prepare(_benchmark_path):
        worktree.mkdir()
        return worktree

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner, "_prepare_historical_baseline_worktree", prepare)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="historical run failed"), runner._historical_baseline_worktree_context("benchmark.json"):
        raise RuntimeError("historical run failed")

    assert not worktree.exists()
    assert any(command[:3] == ["git", "worktree", "remove"] for command in calls)
