import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_action_chunking_real_benchmark as runner

TASK_IDS = [f"T-{index:02d}" for index in range(20)]
PILOT_TASK_IDS = list(runner.PILOT_TASK_IDS)
BASE_REPRODUCIBILITY = {
    "model_name": "gpt-5.6-luna",
    "model_version": "openai-compatible-responses",
    "decoding": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 768},
    "task_ids": TASK_IDS,
    "task_prompt_snapshot_id": "sha256:prompt-v1",
    "prompt_snapshot_id": "sha256:prompt-v1",
    "fixture_snapshot_id": "sha256:fixture-v1",
    "step_budget": 8,
    "step_budget_summary": {"min": 8, "max": 8, "unique": [8], "count": 20},
    "execution_config": {
        "continue_on_error": False,
        "allowed_tools_by_task": {
            task_id: ["list_files", "read_file", "search"] for task_id in TASK_IDS
        },
    },
}
EXPECTED_CONTRACT = {
    "fixture_snapshot_id": "sha256:fixture-v1",
    "task_prompt_snapshot_id": "sha256:prompt-20-v1",
    "step_budget_summary": {"min": 8, "max": 8, "unique": [8], "count": 20},
}


def _row(task_id, group, *, usage=True):
    is_baseline = group == "A"
    attempt = {
        "input_tokens": 100 if usage else None,
        "output_tokens": 10 if usage else None,
        "total_tokens": 110 if usage else None,
        "cached_tokens": 20 if usage else None,
    }
    return {
        "id": task_id,
        "category": "known_path_multi_file_inspection",
        "status": "pass",
        "passed": True,
        "verifier_passed": True,
        "failure_category": None,
        "logical_decisions": 10 if is_baseline else 8,
        "provider_requests": 10 if is_baseline else 8,
        "input_tokens": attempt["input_tokens"],
        "output_tokens": attempt["output_tokens"],
        "total_tokens": attempt["total_tokens"],
        "cached_tokens": attempt["cached_tokens"],
        "provider_attempts": [attempt],
        "chunk_count": 0 if is_baseline else 1,
        "chunk_lengths": [] if is_baseline else [2],
        "chunk_interrupt_rate": 0.0,
        "e2e_latency_ms": 10,
    }


def _write_artifact(path, group, repetition, *, commit="commit-x", task_ids=None, usage=True):
    task_ids = list(task_ids or TASK_IDS)
    reproducibility = deepcopy(BASE_REPRODUCIBILITY)
    reproducibility["task_ids"] = task_ids
    reproducibility["task_prompt_snapshot_id"] = f"sha256:prompt-{len(task_ids)}-v1"
    reproducibility["prompt_snapshot_id"] = reproducibility["task_prompt_snapshot_id"]
    reproducibility["step_budget_summary"] = {
        "min": 8,
        "max": 8,
        "unique": [8],
        "count": len(task_ids),
    }
    reproducibility["execution_config"] = {
        "continue_on_error": False,
        "allowed_tools_by_task": {
            task_id: ["list_files", "read_file", "search"] for task_id in task_ids
        },
    }
    reproducibility["action_chunking"] = runner.normalize_action_chunking(runner.GROUP_CONFIGS[group])
    payload = {
        "schema_version": 1,
        "artifact_id": f"{group}-rep-{repetition:02d}",
        "group": group,
        "repetition": repetition,
        "runtime": {"commit_sha": commit},
        "reproducibility": reproducibility,
        "rows": [_row(task_id, group, usage=usage) for task_id in task_ids],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _artifact_set(tmp_path, *, repetitions=3, commits=None, task_ids=None, usage=True):
    task_ids = list(task_ids or TASK_IDS)
    commits = commits or {group: "commit-x" for group in ("A", "B", "C")}
    artifact_paths = {}
    for group in ("A", "B", "C"):
        artifact_paths[group] = [
            _write_artifact(
                tmp_path / group / f"rep-{repetition:02d}.json",
                group,
                repetition,
                commit=commits[group],
                task_ids=task_ids,
                usage=usage,
            )
            for repetition in range(1, repetitions + 1)
        ]
    return artifact_paths


def _summary(artifact_paths, *, task_ids=TASK_IDS, mode="final", expected_contract=None):
    kwargs = {"mode": mode}
    if expected_contract is not None:
        kwargs["expected_contract"] = expected_contract
    return runner.summarize_real_artifacts(artifact_paths, task_ids, **kwargs)


def _set_correctness(artifact_paths, group, percentage):
    for path in artifact_paths[group]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        passed_count = round(len(payload["rows"]) * percentage / 100)
        for index, row in enumerate(payload["rows"]):
            passed = index < passed_count
            row["passed"] = passed
            row["verifier_passed"] = passed
            row["status"] = "pass" if passed else "fail"
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_final_same_commit_is_hard_gate_and_consistent_commits_pass(tmp_path):
    mismatched = _artifact_set(tmp_path / "mismatched", commits={"A": "X", "B": "Y", "C": "Y"})
    invalid = _summary(mismatched)

    assert invalid["gates"]["same_commit_across_groups"] is False
    assert invalid["final_conclusion"] != "PASS"

    consistent = _summary(_artifact_set(tmp_path / "consistent", commits={"A": "X", "B": "X", "C": "X"}))
    assert consistent["gates"]["same_commit_across_groups"] is True
    assert consistent["final_conclusion"] == "PASS"


def test_rendered_same_commit_status_comes_from_artifacts(tmp_path):
    artifact_paths = _artifact_set(
        tmp_path,
        commits={"A": "commit-a", "B": "commit-b", "C": "commit-b"},
    )
    summary = _summary(artifact_paths)
    report = runner.render_report(
        summary,
        {
            "commit_sha": "environment-commit",
            "model": "gpt-5.6-luna",
            "repetitions": 3,
        },
    )

    assert "Same commit across A/B/C: `FAIL`" in report
    assert "Same commit across A/B/C: `PASS`" not in report


@pytest.mark.parametrize(
    ("counts", "baseline_expected", "b_expected", "c_expected", "status_expected"),
    [
        ((0, 0, 0), False, True, True, False),
        ((85, 80, 85), True, True, True, True),
        ((100, 90, 90), True, False, False, False),
    ],
)
def test_final_correctness_uses_absolute_and_relative_thresholds(
    tmp_path,
    counts,
    baseline_expected,
    b_expected,
    c_expected,
    status_expected,
):
    artifact_paths = _artifact_set(tmp_path)
    for group, count in zip(("A", "B", "C"), counts):
        _set_correctness(artifact_paths, group, count)

    summary = _summary(artifact_paths)

    assert summary["gates"]["baseline_correctness_A_ge_80pct"] is baseline_expected
    assert summary["gates"]["correctness_B_within_5pp"] is b_expected
    assert summary["gates"]["correctness_C_within_5pp"] is c_expected
    assert (summary["final_conclusion"] == "PASS") is status_expected
    if counts == (0, 0, 0):
        assert summary["final_conclusion"] == "INCONCLUSIVE"
    elif counts == (100, 90, 90):
        assert summary["final_conclusion"] == "NEEDS_REVISION"


def test_annotate_primary_artifact_preserves_observed_contract_fields():
    observed = {
        "fixture_snapshot_id": "observed-fixture",
        "task_prompt_snapshot_id": "observed-prompt",
        "step_budget_summary": {"min": 7, "max": 9, "unique": [7, 8, 9], "count": 3},
    }
    artifact = {"reproducibility": deepcopy(observed)}
    environment = deepcopy(BASE_REPRODUCIBILITY)
    environment.update(
        {
            "commit_sha": "environment-commit",
            "model": "gpt-5.6-luna",
            "step_budget": 8,
            "task_prompt_snapshot_id": "expected-prompt",
            "fixture_snapshot_id": "expected-fixture",
            "step_budget_summary": {"min": 8, "max": 8, "unique": [8], "count": 20},
        }
    )

    runner._annotate_primary_artifact(
        artifact,
        "A",
        1,
        TASK_IDS,
        environment,
    )

    for field in ("fixture_snapshot_id", "task_prompt_snapshot_id", "step_budget_summary"):
        assert artifact["reproducibility"][field] == observed[field]


def test_expected_contract_matching_snapshots_pass(tmp_path):
    matching = _summary(_artifact_set(tmp_path), expected_contract=EXPECTED_CONTRACT)

    assert matching["experimental_validity"]["matches_expected_prompt_snapshot"] is True
    assert matching["experimental_validity"]["matches_expected_fixture_snapshot"] is True
    assert matching["experimental_validity"]["matches_expected_step_budget"] is True
    assert matching["gates"]["experimental_validity"] is True
    assert matching["final_conclusion"] == "PASS"


@pytest.mark.parametrize(
    ("field", "matches_field", "mismatch"),
    [
        ("task_prompt_snapshot_id", "matches_expected_prompt_snapshot", "wrong-prompt"),
        ("fixture_snapshot_id", "matches_expected_fixture_snapshot", "wrong-fixture"),
        (
            "step_budget_summary",
            "matches_expected_step_budget",
            {"min": 7, "max": 7, "unique": [7], "count": 20},
        ),
    ],
)
def test_expected_contract_snapshot_mismatches_fail_closed(
    tmp_path,
    field,
    matches_field,
    mismatch,
):
    artifact_paths = _artifact_set(tmp_path)

    payload = json.loads(artifact_paths["B"][0].read_text(encoding="utf-8"))
    payload["reproducibility"][field] = mismatch
    artifact_paths["B"][0].write_text(json.dumps(payload), encoding="utf-8")
    invalid = _summary(artifact_paths, expected_contract=EXPECTED_CONTRACT)

    assert invalid["experimental_validity"][matches_field] is False
    assert invalid["gates"]["experimental_validity"] is False
    assert invalid["final_conclusion"] != "PASS"


def test_positive_three_repetition_fixtures_include_real_artifact_metadata(tmp_path):
    artifacts = _artifact_set(tmp_path)

    for group in ("A", "B", "C"):
        for path in artifacts[group]:
            reproducibility = json.loads(path.read_text(encoding="utf-8"))["reproducibility"]
            assert reproducibility["model_name"]
            assert reproducibility["model_version"]
            assert reproducibility["decoding"]
            assert reproducibility["task_ids"] == TASK_IDS
            assert reproducibility["task_prompt_snapshot_id"]
            assert reproducibility["fixture_snapshot_id"]
            assert reproducibility["step_budget_summary"]["count"] == len(TASK_IDS)
            assert reproducibility["execution_config"]
            assert reproducibility["action_chunking"] == runner.normalize_action_chunking(runner.GROUP_CONFIGS[group])


@pytest.mark.parametrize("treatment_case", ["missing", "wrong", "same"])
def test_final_requires_explicit_fixed_action_chunking_treatment_matrix(tmp_path, treatment_case):
    artifact_paths = _artifact_set(tmp_path)
    if treatment_case == "missing":
        payload = json.loads(artifact_paths["C"][0].read_text(encoding="utf-8"))
        del payload["reproducibility"]["action_chunking"]
        artifact_paths["C"][0].write_text(json.dumps(payload), encoding="utf-8")
    elif treatment_case == "wrong":
        for group in ("A", "B", "C"):
            for path in artifact_paths[group]:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["reproducibility"]["action_chunking"] = runner.normalize_action_chunking(
                    runner.GROUP_CONFIGS["A"]
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        for group in ("B", "C"):
            for path in artifact_paths[group]:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["reproducibility"]["action_chunking"] = runner.normalize_action_chunking(
                    runner.GROUP_CONFIGS["A"]
                )
                path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _summary(artifact_paths)

    assert summary["experimental_validity"]["action_chunking_unique_treatment"] is False
    assert summary["gates"]["experimental_validity"] is False
    assert summary["final_conclusion"] != "PASS"


@pytest.mark.parametrize("repetitions", [1, 2])
def test_final_requires_at_least_three_repetitions_from_artifacts(tmp_path, repetitions):
    summary = _summary(_artifact_set(tmp_path, repetitions=repetitions))

    assert summary["gates"]["repetition_integrity"] is False
    assert summary["final_conclusion"] != "PASS"


def test_final_repetition_integrity_rejects_incomplete_or_duplicate_artifacts(tmp_path):
    complete = _artifact_set(tmp_path / "complete")
    complete_summary = _summary(complete)
    assert complete_summary["gates"]["repetition_integrity"] is True
    assert complete_summary["final_conclusion"] == "PASS"

    missing_group = _artifact_set(tmp_path / "missing-group")
    del missing_group["C"]
    missing_group_summary = _summary(missing_group)
    assert missing_group_summary["gates"]["repetition_integrity"] is False
    assert missing_group_summary["final_conclusion"] != "PASS"

    missing_task = _artifact_set(tmp_path / "missing-task")
    _write_artifact(
        missing_task["B"][1],
        "B",
        2,
        task_ids=TASK_IDS[:-1],
    )
    missing_task_summary = _summary(missing_task)
    assert missing_task_summary["gates"]["repetition_integrity"] is False
    assert missing_task_summary["final_conclusion"] != "PASS"

    duplicate_artifact = _artifact_set(tmp_path / "duplicate-artifact")
    duplicate_artifact["C"] = [duplicate_artifact["C"][0], *duplicate_artifact["C"]]
    duplicate_summary = _summary(duplicate_artifact)
    assert duplicate_summary["gates"]["repetition_integrity"] is False
    assert duplicate_summary["final_conclusion"] != "PASS"


@pytest.mark.parametrize("duplicate_identity", ["artifact_id", "repetition"])
def test_final_repetition_integrity_rejects_different_files_with_duplicate_identity(
    tmp_path,
    duplicate_identity,
):
    artifact_paths = _artifact_set(tmp_path)
    original = artifact_paths["C"][0]
    duplicate = tmp_path / "C" / f"rep-03-{duplicate_identity}.json"
    payload = json.loads(artifact_paths["C"][2].read_text(encoding="utf-8"))
    payload["rows"][0]["e2e_latency_ms"] = 11
    if duplicate_identity == "artifact_id":
        payload["artifact_id"] = json.loads(original.read_text(encoding="utf-8"))["artifact_id"]
    else:
        payload = json.loads(original.read_text(encoding="utf-8"))
        payload["artifact_id"] = "C-rep-04-unique"
        payload["repetition"] = 1
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    artifact_paths["C"][2] = duplicate

    assert len(artifact_paths["C"]) == len(artifact_paths["A"]) == len(artifact_paths["B"]) == 3

    summary = _summary(artifact_paths)

    assert summary["gates"]["repetition_integrity"] is False
    assert summary["final_conclusion"] != "PASS"


@pytest.mark.parametrize(
    ("field", "value"),
    [("artifact_id", ["not-a-string"]), ("repetition", {"number": 1})],
)
def test_final_repetition_integrity_rejects_malformed_identity_without_raising(
    tmp_path,
    field,
    value,
):
    artifact_paths = _artifact_set(tmp_path)
    target = artifact_paths["C"][0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[field] = value
    target.write_text(json.dumps(payload), encoding="utf-8")

    summary = _summary(artifact_paths)

    assert summary["gates"]["repetition_integrity"] is False
    assert summary["final_conclusion"] != "PASS"


class _MissingUsageFake(runner.FakeModelClient):
    def complete(self, prompt, max_new_tokens, **kwargs):
        response = super().complete(prompt, max_new_tokens, **kwargs)
        self.last_provider_attempts = [
            {
                **attempt,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_tokens": None,
            }
            for attempt in self.last_provider_attempts
        ]
        return response


def test_missing_usage_keeps_provider_and_final_core_gates_eligible(tmp_path):
    clients = iter(
        [
            _MissingUsageFake(["PICO_RESPONSES_PREFLIGHT_OK"]),
            _MissingUsageFake(
                [
                    '<tool>{"name":"read_file","args":{"path":"app/config.py","start":1,"end":20}}</tool>',
                    "<final>The timeout is 30.</final>",
                ]
            ),
            _MissingUsageFake(
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
    preflight = runner._preflight(
        "fake-key",
        "https://api.example.invalid/v1",
        1,
        client_factory=lambda: next(clients),
    )

    assert preflight["provider"]["response_match"] is True
    assert preflight["provider"]["token_usage_complete"] is False
    assert preflight["status"] == "passed"
    assert preflight["pico_tool_smoke"]["status"] == "passed"
    assert preflight["pico_chunk_smoke"]["status"] == "passed"

    summary = _summary(_artifact_set(tmp_path, usage=False))
    assert summary["gates"]["token_metric_valid"] is False
    assert summary["comparisons"]["B_vs_A"]["input_tokens"]["reduction_pct"] is None
    assert summary["final_conclusion"] == "PASS"


def test_pilot_requires_exactly_one_repetition_before_chunk_and_correctness_gates(tmp_path):
    two_repetitions = _summary(
        _artifact_set(tmp_path / "two", repetitions=2, task_ids=PILOT_TASK_IDS),
        task_ids=PILOT_TASK_IDS,
        mode="pilot",
    )
    assert two_repetitions["pilot_gates"]["task_count_is_10"] is True
    assert two_repetitions["pilot_status"] == "FAIL"

    one_repetition = _summary(
        _artifact_set(tmp_path / "one", repetitions=1, task_ids=PILOT_TASK_IDS),
        task_ids=PILOT_TASK_IDS,
        mode="pilot",
    )
    assert one_repetition["pilot_gates"]["task_count_is_10"] is True
    assert one_repetition["pilot_gates"]["A_pass_rate_ge_80pct"] is True
    assert one_repetition["pilot_gates"]["coverage_B_ge_70pct"] is True
    assert one_repetition["pilot_gates"]["coverage_C_ge_70pct"] is True
    assert one_repetition["pilot_status"] == "PASS"


def test_pilot_rejects_single_artifacts_with_non_one_repetition_metadata(tmp_path):
    artifact_paths = _artifact_set(tmp_path, repetitions=1, task_ids=PILOT_TASK_IDS)
    for path in artifact_paths["A"] + artifact_paths["B"] + artifact_paths["C"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["repetition"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _summary(artifact_paths, task_ids=PILOT_TASK_IDS, mode="pilot")

    assert summary["artifact_integrity"]["exact_one_repetition"] is False
    assert summary["pilot_gates"]["exactly_one_repetition"] is False
    assert summary["pilot_status"] == "FAIL"


def test_pilot_requires_experimental_validity_even_when_protocol_coverage_passes(tmp_path):
    artifact_paths = _artifact_set(tmp_path, repetitions=1, task_ids=PILOT_TASK_IDS)
    for path in artifact_paths["B"] + artifact_paths["C"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["reproducibility"]["action_chunking"] = runner.normalize_action_chunking(
            runner.GROUP_CONFIGS["A"]
        )
        path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _summary(artifact_paths, task_ids=PILOT_TASK_IDS, mode="pilot")

    assert summary["pilot_gates"]["coverage_B_ge_70pct"] is True
    assert summary["pilot_gates"]["coverage_C_ge_70pct"] is True
    assert summary["experimental_validity"]["action_chunking_unique_treatment"] is False
    assert summary["pilot_gates"]["experimental_validity"] is False
    assert summary["pilot_status"] == "FAIL"


@pytest.mark.parametrize(
    ("counts", "b_expected", "c_expected", "status_expected"),
    [
        ((100, 0, 0), False, False, False),
        ((100, 90, 90), False, False, False),
        ((90, 90, 90), True, True, True),
    ],
)
def test_pilot_uses_relative_b_c_correctness_not_chunk_or_protocol_coverage(
    tmp_path,
    counts,
    b_expected,
    c_expected,
    status_expected,
):
    artifact_paths = _artifact_set(tmp_path, repetitions=1, task_ids=PILOT_TASK_IDS)
    for group, count in zip(("A", "B", "C"), counts):
        _set_correctness(artifact_paths, group, count)

    summary = _summary(artifact_paths, task_ids=PILOT_TASK_IDS, mode="pilot")

    assert summary["pilot_gates"]["coverage_B_ge_70pct"] is True
    assert summary["pilot_gates"]["coverage_C_ge_70pct"] is True
    assert summary["pilot_gates"]["experimental_validity"] is True
    assert summary["pilot_gates"]["correctness_B_within_5pp"] is b_expected
    assert summary["pilot_gates"]["correctness_C_within_5pp"] is c_expected
    assert (summary["pilot_status"] == "PASS") is status_expected


def test_rendered_treatment_line_is_dynamic_for_invalid_and_valid_experiments(tmp_path):
    invalid_artifacts = _artifact_set(tmp_path / "invalid")
    for path in invalid_artifacts["C"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["reproducibility"]["action_chunking"] = runner.normalize_action_chunking(
            runner.GROUP_CONFIGS["A"]
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
    valid_artifacts = _artifact_set(tmp_path / "valid")

    def treatment_line(summary):
        report = runner.render_report(
            summary,
            {"commit_sha": "commit-x", "model": "gpt-5.6-luna", "repetitions": 3},
        )
        lines = [
            line
            for line in report.splitlines()
            if "Action Chunking is the only treatment" in line
        ]
        assert len(lines) == 1
        return lines[0]

    invalid_line = treatment_line(_summary(invalid_artifacts))
    valid_line = treatment_line(_summary(valid_artifacts))
    assert invalid_line == "- Action Chunking is the only treatment: `FAIL`"
    assert valid_line == "- Action Chunking is the only treatment: `PASS`"


def _run_fixture_verifier(fixture, command):
    return subprocess.run(
        [sys.executable, "-c", command],
        cwd=fixture,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_fixture_report(fixture, answer, paths):
    run_dir = Path(fixture) / ".pico" / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stop_reason": "final_answer_returned",
                "final_answer": answer,
                "task_state": {"primitive_submissions": len(paths)},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps({"event": "tool_executed", "args": {"path": path}}) + "\n" for path in paths),
        encoding="utf-8",
    )


def _prepare_request_order_fixture(tmp_path):
    fixture = tmp_path / "fixture"
    shutil.copytree("benchmarks/fixtures/action_chunk_repo", fixture)
    service_path = fixture / "app" / "service.py"
    source = service_path.read_text(encoding="utf-8")
    source = """
from app.formatter import validate_request, route_request, get_or_put, format_response

# Noise comment: format_response, get_or_put, route_request, validate_request.
def unrelated_helper(request):
    # Same symbols in an unrelated function must not determine request order.
    format_response(request)
    get_or_put(request)
    route_request(request)
    validate_request(request)

""" + source
    function_body = """
def handle_request(request, cache=None):
    \"\"\"Validate, route, and format one request.\"\"\"
    validate_request(request)
    route = route_request(request)
    cache = cache or CacheStore()
    result = cache.get_or_put(route, lambda: {\"route\": route, \"ok\": True})
    return format_response(result)
"""
    shuffled_body = """
def handle_request(request, cache=None):
    \"\"\"Intentionally shuffled for the verifier regression test.\"\"\"
    result = get_or_put(request)
    return format_response(result)
    route = route_request(request)
    validate_request(request)
"""
    return fixture, service_path, source, function_body, shuffled_body


def _request_order_command():
    return (
        "from verify_benchmark import check_semantic; "
        "check_semantic([], 1, paths=['app/service.py'], facts=['request_order'])"
    )


def test_request_order_uses_handle_request_positions_not_symbol_presence(tmp_path):
    fixture, service_path, source, correct_body, shuffled_body = _prepare_request_order_fixture(tmp_path)
    answer = "validate_request route_request get_or_put format_response"
    _write_fixture_report(fixture, answer, ["app/service.py"])

    service_path.write_text(source.replace(correct_body, shuffled_body), encoding="utf-8")
    shuffled_result = _run_fixture_verifier(fixture, _request_order_command())
    assert shuffled_result.returncode != 0

    service_path.write_text(source, encoding="utf-8")
    correct_result = _run_fixture_verifier(fixture, _request_order_command())
    assert correct_result.returncode == 0, correct_result.stderr


@pytest.mark.parametrize(
    ("path", "validity_key"),
    [
        (("reproducibility", "model_name"), "same_model"),
        (("reproducibility", "model_version"), "same_model"),
        (("reproducibility", "decoding", "temperature"), "same_decoding"),
        (("reproducibility", "decoding", "max_new_tokens"), "same_decoding"),
        (("reproducibility", "task_ids"), "same_task_ids"),
        (("reproducibility", "task_prompt_snapshot_id"), "same_prompt_snapshot"),
        (("reproducibility", "fixture_snapshot_id"), "same_fixture_snapshot"),
        (("reproducibility", "step_budget_summary"), "same_step_budget"),
        (("reproducibility", "execution_config", "continue_on_error"), "same_execution_config"),
        (("runtime", "commit_sha"), "same_commit"),
    ],
)
def test_cross_artifact_experimental_validity_is_derived_from_artifacts(
    tmp_path,
    path,
    validity_key,
):
    artifact_paths = _artifact_set(tmp_path)
    baseline = _summary(artifact_paths)
    assert baseline["experimental_validity"][validity_key] is True
    assert baseline["final_conclusion"] == "PASS"

    target = artifact_paths["B"][0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    field = path[-1]
    cursor[field] = (
        "different-value"
        if isinstance(cursor[field], str)
        else (cursor[field] + 1 if isinstance(cursor[field], (int, float)) else ["different-task"])
    )
    target.write_text(json.dumps(payload), encoding="utf-8")

    invalid = _summary(artifact_paths)
    assert invalid["experimental_validity"][validity_key] is False
    assert invalid["final_conclusion"] != "PASS"


def test_worktree_cleanup_prunes_stale_metadata_after_remove_failure(tmp_path, monkeypatch):
    stale_worktree = tmp_path / "historical-worktree"
    other_worktree = tmp_path / "user-worktree"
    stale_worktree.mkdir()
    other_worktree.mkdir()
    (other_worktree / "keep.txt").write_text("keep", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(list(command))
        if command[:3] == ["git", "worktree", "remove"]:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "stale metadata"})()
        if command[:3] == ["git", "worktree", "prune"]:
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._cleanup_historical_baseline_worktree(stale_worktree)

    assert not stale_worktree.exists()
    assert (other_worktree / "keep.txt").read_text(encoding="utf-8") == "keep"
    remove_calls = [command for command in calls if command[:3] == ["git", "worktree", "remove"]]
    assert remove_calls == [["git", "worktree", "remove", "--force", str(stale_worktree)]]
    assert calls.index(["git", "worktree", "prune"]) > calls.index(remove_calls[0])


def test_worktree_cleanup_remove_success_cleans_only_target(tmp_path, monkeypatch):
    stale_worktree = tmp_path / "historical-worktree"
    sibling = tmp_path / "sibling-worktree"
    stale_worktree.mkdir()
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(list(command))
        assert command[:3] == ["git", "worktree", "remove"]
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._cleanup_historical_baseline_worktree(stale_worktree)

    assert not stale_worktree.exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert calls == [["git", "worktree", "remove", "--force", str(stale_worktree)]]


def test_worktree_cleanup_prune_failure_still_cleans_target_and_sibling_is_untouched(
    tmp_path,
    monkeypatch,
):
    stale_worktree = tmp_path / "historical-worktree"
    sibling = tmp_path / "sibling-worktree"
    stale_worktree.mkdir()
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(list(command))
        if command[:3] == ["git", "worktree", "remove"]:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "remove failed"})()
        if command[:3] == ["git", "worktree", "prune"]:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "prune failed"})()
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._cleanup_historical_baseline_worktree(stale_worktree)

    assert not stale_worktree.exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert calls == [
        ["git", "worktree", "remove", "--force", str(stale_worktree)],
        ["git", "worktree", "prune"],
    ]
