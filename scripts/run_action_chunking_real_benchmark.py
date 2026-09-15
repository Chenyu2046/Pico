"""Run the fixed A/B/C Action Chunking benchmark.

The runner keeps credentials in the process environment and writes only
non-secret experiment metadata.  Real-model results are meaningful only when
the provider preflight succeeds and the final gates pass.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pico.action_chunk import normalize_action_chunking
from pico.evaluation.evaluator import (
    DEFAULT_TIMEZONE,
    BenchmarkEvaluator,
    _current_locale,
    _fixture_snapshot_id,
    _git_value,
    aggregate_provider_usage,
    load_benchmark,
    run_fixed_benchmark,
)
from pico.providers.clients import FakeModelClient, OpenAICompatibleModelClient
from pico.run_store import RunStore
from pico.runtime import Pico, SessionStore
from pico.task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from pico.workspace import WorkspaceContext

MODEL_NAME = "gpt-5.6-luna"
DEFAULT_BASE_URL = "https://api.longxiadev.store/v1"
DEFAULT_OUTPUT_DIR = Path("artifacts/action-chunking-real")
ALLOWED_TOOLS = ["list_files", "read_file", "search"]
GROUP_CONFIGS = {
    "A": {"enabled": False},
    "B": {
        "enabled": True,
        "max_actions_per_chunk": 4,
        "allowed_tools": ALLOWED_TOOLS,
        "observation_budget_chars": 1_000_000,
        "skill_guidance_enabled": False,
    },
    "C": {
        "enabled": True,
        "max_actions_per_chunk": 4,
        "allowed_tools": ALLOWED_TOOLS,
        "observation_budget_chars": 12_000,
        "skill_guidance_enabled": False,
    },
}
PILOT_TASK_IDS = [
    "MF-01", "MF-02", "MF-03", "MF-04", "MF-05", "MF-06",
    "OB-01", "OB-02", "SC-01", "SC-02",
]
ROTATIONS = (
    ("A", "B", "C"),
    ("B", "C", "A"),
    ("C", "A", "B"),
    ("A", "C", "B"),
    ("C", "B", "A"),
)
POSITIVE_CATEGORIES = {
    "known_path_multi_file_inspection",
    "multi_symbol_search",
    "config_impl_test_inspection",
}
ORIGINAL_BASELINE_SHA = "6e3012f12b9c24c99bce17b9fe373534290e5ebb"


def _json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_error(exc, secret=""):
    text = str(exc)
    if secret:
        text = text.replace(secret, "<redacted>")
    return text[:1000]


def _redact(value, secret):
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _task_prompt_snapshot_id(tasks, task_ids):
    prompts = {
        task["id"]: str(task["prompt"])
        for task in tasks
        if task["id"] in set(task_ids)
    }
    payload = "\n".join(f"{task_id}\0{prompts[task_id]}" for task_id in task_ids)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cleanup_historical_baseline_worktree(worktree):
    if worktree is None:
        return
    worktree = Path(worktree)
    remove_failed = False
    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        remove_failed = result.returncode != 0
    except OSError:
        remove_failed = True
    finally:
        shutil.rmtree(worktree, ignore_errors=True)
        if remove_failed:
            try:
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError:
                pass


def _prepare_historical_baseline_worktree(benchmark_path):
    worktree = Path(tempfile.mkdtemp(prefix="pico-action-chunk-baseline-"))
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), ORIGINAL_BASELINE_SHA],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        _cleanup_historical_baseline_worktree(worktree)
        raise RuntimeError("could not create original baseline worktree: " + _safe_error(result.stderr))

    try:
        benchmark_target = worktree / "benchmarks" / "action_chunking_tasks.json"
        benchmark_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(benchmark_path, benchmark_target)
        fixture_target = worktree / "benchmarks" / "fixtures" / "action_chunk_repo"
        fixture_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO_ROOT / "benchmarks" / "fixtures" / "action_chunk_repo", fixture_target)
        helper_target = worktree / "scripts" / "run_action_chunking_baseline.py"
        helper_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "scripts" / "run_action_chunking_baseline.py", helper_target)
        usage_target = worktree / "pico" / "evaluation" / "token_usage.py"
        usage_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "pico" / "evaluation" / "token_usage.py", usage_target)
        return worktree
    except Exception:
        _cleanup_historical_baseline_worktree(worktree)
        raise


@contextmanager
def _historical_baseline_worktree_context(benchmark_path):
    worktree = None
    try:
        worktree = _prepare_historical_baseline_worktree(benchmark_path)
        yield worktree
    finally:
        _cleanup_historical_baseline_worktree(worktree)


def _run_baseline_task_set(worktree, artifact_path, workspace_root, task_ids, api_key, base_url, temperature, max_new_tokens, timeout):
    command = [
        sys.executable,
        str(worktree / "scripts" / "run_action_chunking_baseline.py"),
        "--benchmark", str(worktree / "benchmarks" / "action_chunking_tasks.json"),
        "--output", str(artifact_path),
        "--workspace", str(workspace_root),
        "--base-url", base_url,
        "--model", MODEL_NAME,
        "--temperature", str(temperature),
        "--max-new-tokens", str(max_new_tokens),
        "--timeout", str(timeout),
        "--tasks", ",".join(task_ids),
    ]
    environment = dict(os.environ)
    environment["PICO_OPENAI_API_KEY"] = api_key
    result = subprocess.run(command, cwd=worktree, env=environment, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("original baseline runner failed: " + _safe_error(result.stderr, api_key))
    return json.loads(Path(artifact_path).read_text(encoding="utf-8"))


def _run_historical_reference(
    worktree,
    run_root,
    task_ids,
    api_key,
    base_url,
    temperature,
    max_new_tokens,
    timeout,
):
    artifact_path = Path(run_root) / "historical" / f"baseline-{ORIGINAL_BASELINE_SHA[:12]}.json"
    artifact = _run_baseline_task_set(
        worktree,
        artifact_path,
        Path(run_root) / "workspaces" / "historical",
        task_ids,
        api_key,
        base_url,
        temperature,
        max_new_tokens,
        timeout,
    )
    return {
        "status": "passed",
        "commit_sha": ORIGINAL_BASELINE_SHA,
        "artifact": str(artifact_path),
        "summary": artifact.get("summary", {}),
        "not_used_in_primary_comparisons": True,
    }


def _run_compatibility_regression(baseline_worktree, run_root):
    before_path = Path(run_root) / "compatibility" / "before.json"
    after_path = Path(run_root) / "compatibility" / "after.json"
    after = run_fixed_benchmark(
        benchmark_path=REPO_ROOT / "benchmarks" / "coding_tasks.json",
        artifact_path=after_path,
        workspace_root=Path(run_root) / "compatibility" / "after-workspaces",
        action_chunking={"enabled": False},
    )
    baseline_benchmark = baseline_worktree / "benchmarks" / "coding_tasks.json"
    baseline_code = (
        "from pathlib import Path; "
        "from pico.evaluation.evaluator import run_fixed_benchmark; "
        f"run_fixed_benchmark(benchmark_path=Path({str(baseline_benchmark)!r}), "
        f"artifact_path=Path({str(before_path)!r}), "
        f"workspace_root=Path({str(Path(run_root) / 'compatibility' / 'before-workspaces')!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", baseline_code],
        cwd=baseline_worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("compatibility baseline runner failed: " + _safe_error(result.stderr))
    before = json.loads(before_path.read_text(encoding="utf-8"))

    def evidence(artifact, commit_sha):
        rows = artifact.get("rows", [])
        return {
            "commit_sha": commit_sha,
            "passed": artifact.get("summary", {}).get("passed", 0),
            "total_tasks": artifact.get("summary", {}).get("total_tasks", len(rows)),
            "all_chunk_count_zero": all(row.get("chunk_count", 0) == 0 for row in rows),
        }

    return {
        "before": evidence(before, ORIGINAL_BASELINE_SHA),
        "after": evidence(after, _git_value(["rev-parse", "HEAD"], cwd=REPO_ROOT)),
    }


def _tool(name, args):
    return "<tool>" + json.dumps({"name": name, "args": args}, separators=(",", ":")) + "</tool>"


def _chunk(actions):
    actions = [
        json.loads(action[len("<tool>"):-len("</tool>")]) if isinstance(action, str) else action
        for action in actions
    ]
    return "<chunk>" + json.dumps({"actions": actions}, separators=(",", ":")) + "</chunk>"


def _planned_actions(task):
    task_id = task["id"]
    workload = task.get("workload", {})
    if task_id == "SC-01":
        return [
            _tool("read_file", {"path": "README.md", "start": 1, "end": 80}),
            _tool("search", {"pattern": "route_request", "path": "app"}),
            _tool("read_file", {"path": "app/router.py", "start": 1, "end": 80}),
            _tool("read_file", {"path": "app/service.py", "start": 1, "end": 80}),
        ]
    if task_id == "SC-02":
        return [
            _tool("search", {"pattern": "CacheStore", "path": "app"}),
            _tool("read_file", {"path": "app/cache.py", "start": 1, "end": 80}),
            _tool("read_file", {"path": "app/service.py", "start": 1, "end": 80}),
        ]
    if task_id == "SC-03":
        return [
            _tool("search", {"pattern": "validate_request", "path": "app"}),
            _tool("read_file", {"path": "app/validator.py", "start": 1, "end": 80}),
            _tool("read_file", {"path": "app/service.py", "start": 1, "end": 80}),
        ]
    if workload.get("expected_paths"):
        actions = []
        for path in workload["expected_paths"]:
            end = 200 if path.endswith("catalog.py") else 80
            actions.append(_tool("read_file", {"path": path, "start": 1, "end": end}))
        return actions
    actions = [
        _tool(
            "search",
            {"pattern": pattern, "path": "tests" if pattern.startswith("test_") else "app"},
        )
        for pattern in workload.get("expected_patterns", [])
    ]
    actions.extend(
        _tool("read_file", {"path": path, "start": 1, "end": 80})
        for path in workload.get("followup_paths", [])
    )
    return actions


def _deterministic_outputs(task, action_chunking):
    actions = _planned_actions(task)
    final = (
        "<final>Inspection complete. Observed facts: DEFAULT_TIMEOUT=30; RETRY_LIMIT=2; "
        "ParseResult fields are method, route, body; the parser normalizes to GET; "
        "the health route maps to health; the response has status 200 and data; "
        "validation reports route must start with /; the request order is "
        "validate_request -> route_request -> get_or_put -> format_response; "
        "load_record records source path; cache reuse uses get, put, and get_or_put; "
        "the fixture marker is deterministic-observation-catalog. Symbols observed: "
        "parse_request, load_record, CacheStore, route_request, handle_request, "
        "validate_request, format_response, test_parse_request_normalizes_method, "
        "test_cache_reuses_value, test_health_route, test_service_formats_health_response, "
        "PICO_TIMEOUT, ROUTE_TABLE, and action-fixture.</final>"
    )
    if not action_chunking.get("enabled"):
        return actions + [final]

    if task.get("category") == "observation_boundary":
        return [_chunk([actions[0]])] + [_chunk(actions[1:]), final]

    if task.get("category") in POSITIVE_CATEGORIES:
        chunks = [_chunk(actions[index:index + 4]) for index in range(0, len(actions), 4)]
        return chunks + [final]

    return actions + [final]


class _DeterministicModelClient(FakeModelClient):
    def complete(self, prompt, max_new_tokens, **kwargs):
        response = super().complete(prompt, max_new_tokens, **kwargs)
        usage = {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "cached_tokens": 20,
        }
        self.last_completion_metadata.update(usage)
        self.last_provider_metadata.update(usage)
        self.last_provider_attempts[-1].update(usage)
        return response


def deterministic_model_factory(action_chunking):
    def factory(task, workspace):
        del workspace
        return _DeterministicModelClient(_deterministic_outputs(task, action_chunking))

    return factory


def _run_pico_smoke(model_client, action_chunking, allowed_tools, prompt):
    temporary_root = Path(tempfile.mkdtemp(prefix="pico-action-chunk-smoke-"))
    fixture_copy = temporary_root / "action_chunk_repo"
    shutil.copytree(REPO_ROOT / "benchmarks" / "fixtures" / "action_chunk_repo", fixture_copy)
    try:
        workspace = WorkspaceContext.build(fixture_copy, repo_root_override=fixture_copy)
        agent = Pico(
            model_client=model_client,
            workspace=workspace,
            session_store=SessionStore(fixture_copy / ".pico" / "sessions"),
            run_store=RunStore(fixture_copy / ".pico" / "runs"),
            approval_policy="auto",
            max_steps=8,
            max_new_tokens=768,
            allowed_tools=allowed_tools,
            action_chunking=action_chunking,
            secret_env_names=["PICO_OPENAI_API_KEY"],
        )
        try:
            agent.ask(prompt)
        except Exception as exc:  # noqa: BLE001 - smoke converts provider/parser errors to evidence
            state = agent.current_task_state
            return {
                "status": "failed",
                "reason": _safe_error(exc),
                "stop_reason": state.stop_reason,
                "primitive_submissions": state.primitive_submissions,
                "chunk_count": state.chunk_count,
                "mean_chunk_length": 0.0,
            }
        state = agent.current_task_state
        mean_chunk_length = (
            sum(state.chunk_lengths) / len(state.chunk_lengths)
            if state.chunk_lengths
            else 0.0
        )
        return {
            "status": "passed" if state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED else "failed",
            "reason": "" if state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED else "smoke did not return a final answer",
            "stop_reason": state.stop_reason,
            "primitive_submissions": state.primitive_submissions,
            "chunk_count": state.chunk_count,
            "mean_chunk_length": mean_chunk_length,
        }
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def run_deterministic_benchmark(
    benchmark_path="benchmarks/action_chunking_tasks.json",
    workspace_root=None,
):
    """Run all 20 tasks through FakeModelClient for contract validation."""
    benchmark_path = Path(benchmark_path).resolve()
    benchmark = load_benchmark(benchmark_path)
    tasks = benchmark["tasks"]
    task_ids = [task["id"] for task in tasks]
    if workspace_root is None:
        workspace_root = tempfile.mkdtemp(prefix="pico-action-chunk-deterministic-")
    result = {}
    for group, config in GROUP_CONFIGS.items():
        artifact_path = Path(workspace_root) / f"deterministic-{group}.json"
        evaluator = BenchmarkEvaluator(
            benchmark_path=benchmark_path,
            artifact_path=artifact_path,
            workspace_root=Path(workspace_root) / f"workspaces-{group}",
            model_name="FakeModelClient",
            model_version="action-chunking-deterministic",
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=768,
            timezone_name=DEFAULT_TIMEZONE,
            model_client_factory=deterministic_model_factory(config),
            action_chunking=config,
        )
        artifact = evaluator.run()
        _annotate_primary_artifact(
            artifact,
            group,
            1,
            task_ids,
            {
                "task_prompt_snapshot_id": _task_prompt_snapshot_id(tasks, task_ids),
                "fixture_snapshot_id": artifact["reproducibility"]["fixture_snapshot_id"],
                "step_budget": benchmark["benchmark_metadata"]["step_budget"],
                "step_budget_summary": artifact["reproducibility"].get("step_budget_summary"),
            },
        )
        _json_write(artifact_path, artifact)
        result[group] = artifact
    return result


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def _distribution(values):
    values = [value for value in values if value is not None]
    return {
        "count": len(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
    }


def _numeric_values(rows, key):
    return [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]


def _aggregate_group(rows):
    rows = list(rows)
    chunk_lengths = [length for row in rows for length in row.get("chunk_lengths", [])]
    positive_rows = [row for row in rows if row.get("category") in POSITIVE_CATEGORIES]
    chunked_positive = [row for row in positive_rows if row.get("chunk_count", 0) > 0]
    positive_chunk_lengths = [
        length for row in positive_rows for length in row.get("chunk_lengths", [])
    ]
    provider_attempts = [
        dict(attempt)
        for row in rows
        for attempt in row.get("provider_attempts", [])
    ]
    token_usage = aggregate_provider_usage(provider_attempts)
    if not provider_attempts and rows:
        token_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "token_usage_coverage": sum(
                float(row.get("token_usage_coverage", 0.0)) for row in rows
            ) / len(rows),
            "token_usage_complete": all(
                bool(row.get("token_usage_complete")) for row in rows
            ),
        }
    return {
        "task_runs": len(rows),
        "passed": sum(bool(row.get("passed")) for row in rows),
        "verifier_pass_rate": (
            sum(bool(row.get("verifier_passed")) for row in rows) / len(rows) if rows else 0.0
        ),
        "pass_rate": sum(bool(row.get("passed")) for row in rows) / len(rows) if rows else 0.0,
        "logical_decisions": _distribution(_numeric_values(rows, "logical_decisions")),
        "provider_requests": _distribution(_numeric_values(rows, "provider_requests")),
        "input_tokens": _distribution(_numeric_values(rows, "input_tokens")),
        "output_tokens": _distribution(_numeric_values(rows, "output_tokens")),
        "total_tokens": _distribution(_numeric_values(rows, "total_tokens")),
        "cached_tokens": _distribution(_numeric_values(rows, "cached_tokens")),
        "e2e_latency_ms": _distribution(_numeric_values(rows, "e2e_latency_ms")),
        "primitive_submissions": _distribution(_numeric_values(rows, "primitive_submissions")),
        "executed_tool_calls": _distribution(_numeric_values(rows, "executed_tool_calls")),
        "successful_tool_calls": _distribution(_numeric_values(rows, "successful_tool_calls")),
        "failed_tool_calls": _distribution(_numeric_values(rows, "failed_tool_calls")),
        "rejected_tool_calls": _distribution(_numeric_values(rows, "rejected_tool_calls")),
        "unknown_tool_calls": _distribution(_numeric_values(rows, "unknown_tool_calls")),
        "chunk_lengths": {
            **_distribution(chunk_lengths),
            "mean": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else None,
        },
        "positive_chunk_lengths": {
            **_distribution(positive_chunk_lengths),
            "mean": (
                sum(positive_chunk_lengths) / len(positive_chunk_lengths)
                if positive_chunk_lengths
                else None
            ),
        },
        "chunk_count": _distribution(_numeric_values(rows, "chunk_count")),
        "token_usage_coverage": token_usage["token_usage_coverage"],
        "token_usage_complete": token_usage["token_usage_complete"],
        "positive_chunk_coverage": len(chunked_positive) / len(positive_rows) if positive_rows else 0.0,
        "positive_task_runs": len(positive_rows),
        "positive_chunked_task_runs": len(chunked_positive),
        "chunk_interrupt_rate": (
            sum(row.get("chunk_interrupt_rate", 0.0) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def _rows_by_task(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)
    return grouped


def _paired_comparison(group_rows, baseline_rows, metric):
    group_by_task = _rows_by_task(group_rows)
    baseline_by_task = _rows_by_task(baseline_rows)
    paired = []
    for task_id in sorted(set(group_by_task) & set(baseline_by_task)):
        group_values = _numeric_values(group_by_task[task_id], metric)
        baseline_values = _numeric_values(baseline_by_task[task_id], metric)
        if group_values and baseline_values:
            group_median = _distribution(group_values)["median"]
            baseline_median = _distribution(baseline_values)["median"]
            paired.append({"task_id": task_id, "baseline": baseline_median, "group": group_median})
    baseline_mean = sum(item["baseline"] for item in paired) / len(paired) if paired else None
    group_mean = sum(item["group"] for item in paired) / len(paired) if paired else None
    reduction = ((baseline_mean - group_mean) / baseline_mean * 100.0) if baseline_mean else None
    return {"task_count": len(paired), "baseline_median_mean": baseline_mean, "group_median_mean": group_mean, "reduction_pct": reduction, "pairs": paired}


def _successful_rows(rows):
    return [row for row in rows if row.get("passed") and row.get("verifier_passed")]


def _protocol_error_count(rows):
    return sum(
        row.get("failure_category") in {"runner_exception", "failure_stop_reason"}
        or row.get("stop_reason") in {"model_error", "action_sequence_mismatch"}
        for row in rows
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _all_equal(values):
    return bool(values) and len({_canonical(value) for value in values}) == 1


def _artifact_integrity(artifact_paths, artifacts, task_ids, mode):
    expected_groups = ("A", "B", "C")
    expected_tasks = list(task_ids)
    expected_task_set = {_canonical(task_id) for task_id in expected_tasks}
    all_paths = []
    artifact_ids = set()
    repetitions = set()
    counts = {}
    complete = True
    duplicate_paths = False
    duplicate_ids = False
    duplicate_repetitions = False

    for group in expected_groups:
        paths = [Path(path) for path in artifact_paths.get(group, [])]
        counts[group] = len(paths)
        group_repetitions = set()
        for path, artifact in zip(paths, artifacts.get(group, [])):
            path_key = str(path.resolve())
            duplicate_paths = duplicate_paths or path_key in all_paths
            all_paths.append(path_key)

            artifact_id = artifact.get("artifact_id")
            repetition = artifact.get("repetition")
            id_key = _canonical(artifact_id)
            repetition_key = (group, _canonical(repetition))
            duplicate_ids = duplicate_ids or id_key in artifact_ids
            duplicate_repetitions = duplicate_repetitions or repetition_key in repetitions
            artifact_ids.add(id_key)
            repetitions.add(repetition_key)
            if not isinstance(artifact_id, str) or not artifact_id:
                complete = False
            if not isinstance(repetition, int) or isinstance(repetition, bool):
                complete = False
            else:
                group_repetitions.add(_canonical(repetition))
            if artifact.get("group") != group:
                complete = False
            row_ids = [_canonical(row.get("id")) for row in artifact.get("rows", [])]
            if len(row_ids) != len(expected_tasks) or set(row_ids) != expected_task_set:
                complete = False
        if group_repetitions != {_canonical(index) for index in range(1, len(paths) + 1)}:
            complete = False

    groups_present = set(artifact_paths) == set(expected_groups)
    counts_equal = len(set(counts.values())) == 1 if counts else False
    min_repetitions = 3 if mode == "final" else 1
    enough_repetitions = counts_equal and all(counts.get(group, 0) >= min_repetitions for group in expected_groups)
    exact_one_repetition = groups_present and counts_equal and all(
        counts.get(group, 0) == 1 for group in expected_groups
    )
    repetition_shape_valid = (
        enough_repetitions if mode == "final" else exact_one_repetition
    )
    repetition_integrity = (
        groups_present
        and counts_equal
        and repetition_shape_valid
        and complete
        and not duplicate_paths
        and not duplicate_ids
        and not duplicate_repetitions
    )
    return {
        "groups_present": groups_present,
        "artifact_counts": counts,
        "artifact_counts_equal": counts_equal,
        "enough_repetitions": enough_repetitions,
        "exact_one_repetition": exact_one_repetition,
        "all_tasks_complete": complete,
        "minimum_repetitions_met": repetition_shape_valid,
        "group_repetition_counts_equal": counts_equal,
        "task_repetition_counts_complete": complete,
        "duplicate_artifact_paths": duplicate_paths,
        "duplicate_artifact_ids": duplicate_ids,
        "duplicate_repetitions": duplicate_repetitions,
        "repetition_integrity": repetition_integrity,
    }


def _experimental_validity(artifacts, task_ids):
    all_artifacts = [artifact for group in ("A", "B", "C") for artifact in artifacts.get(group, [])]
    reproducibility = [artifact.get("reproducibility", {}) for artifact in all_artifacts]
    runtime = [artifact.get("runtime", {}) for artifact in all_artifacts]

    def same_field(items, field):
        values = [item.get(field) for item in items]
        return bool(values) and all(value is not None for value in values) and _all_equal(values)

    def same_reproducibility_field(primary, legacy=None):
        values = [
            item.get(primary) if item.get(primary) is not None else item.get(legacy)
            for item in reproducibility
        ]
        return bool(values) and all(value is not None for value in values) and _all_equal(values)

    same_commit = same_field(runtime, "commit_sha")
    same_model = (
        same_field(reproducibility, "model_name")
        and same_field(reproducibility, "model_version")
    )
    same_decoding = same_field(reproducibility, "decoding")
    same_task_ids = all(
        artifact.get("reproducibility", {}).get("task_ids") == list(task_ids)
        for artifact in all_artifacts
    ) if all_artifacts else False
    same_prompt_snapshot = same_reproducibility_field("task_prompt_snapshot_id", "prompt_snapshot_id")
    same_fixture_snapshot = same_field(reproducibility, "fixture_snapshot_id")
    same_step_budget = same_reproducibility_field("step_budget_summary", "step_budget")
    execution_configs = [item.get("execution_config") for item in reproducibility]
    same_execution_config = (
        bool(execution_configs)
        and all(value is not None for value in execution_configs)
        and same_field(reproducibility, "execution_config")
    )

    common_fields = ("model_name", "model_version", "decoding", "fixture_snapshot_id")
    common_configuration = (
        all(same_field(reproducibility, field) for field in common_fields)
        and same_task_ids
        and same_prompt_snapshot
        and same_step_budget
        and same_execution_config
    )
    action_configs = {
        group: [artifact.get("reproducibility", {}).get("action_chunking") for artifact in artifacts.get(group, [])]
        for group in ("A", "B", "C")
    }
    expected_action_configs = {
        group: normalize_action_chunking(GROUP_CONFIGS[group])
        for group in ("A", "B", "C")
    }
    action_treatment_valid = all(
        values
        and all(_canonical(value) == _canonical(expected_action_configs[group]) for value in values)
        for group, values in action_configs.items()
    )
    action_chunking_unique_treatment = common_configuration and action_treatment_valid
    validity = {
        "same_commit": same_commit,
        "same_model": same_model,
        "same_decoding": same_decoding,
        "same_task_ids": same_task_ids,
        "same_prompt_snapshot": same_prompt_snapshot,
        "same_fixture_snapshot": same_fixture_snapshot,
        "same_step_budget": same_step_budget,
        "same_execution_config": same_execution_config,
        "action_chunking_unique_treatment": action_chunking_unique_treatment,
        "all_required_fields_consistent": all(
            (
                same_commit,
                same_model,
                same_decoding,
                same_task_ids,
                same_prompt_snapshot,
                same_fixture_snapshot,
                same_step_budget,
                same_execution_config,
            )
        ),
    }
    validity["valid"] = validity["all_required_fields_consistent"] and validity["action_chunking_unique_treatment"]
    return validity


def _annotate_primary_artifact(artifact, group, repetition, task_ids, environment):
    artifact["artifact_id"] = f"{group}-rep-{repetition:02d}"
    artifact["group"] = group
    artifact["repetition"] = repetition
    reproducibility = artifact.setdefault("reproducibility", {})
    reproducibility["task_ids"] = list(task_ids)
    reproducibility["task_prompt_snapshot_id"] = environment["task_prompt_snapshot_id"]
    reproducibility["prompt_snapshot_id"] = environment["task_prompt_snapshot_id"]
    reproducibility["fixture_snapshot_id"] = environment["fixture_snapshot_id"]
    reproducibility["step_budget"] = environment["step_budget"]
    if environment.get("step_budget_summary") is not None:
        reproducibility["step_budget_summary"] = environment["step_budget_summary"]
    return artifact


def summarize_real_artifacts(artifact_paths, task_ids, mode="final"):
    artifacts = {
        group: [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
        for group, paths in artifact_paths.items()
    }
    rows = {
        group: [row for artifact in group_artifacts for row in artifact.get("rows", [])]
        for group, group_artifacts in artifacts.items()
    }
    groups = {group: _aggregate_group(group_rows) for group, group_rows in rows.items()}
    metrics = ("logical_decisions", "provider_requests", "input_tokens", "total_tokens", "e2e_latency_ms")
    comparisons = {
        f"{group}_vs_A": {
            metric: _paired_comparison(rows[group], rows["A"], metric)
            for metric in metrics
        }
        for group in ("B", "C")
        if group in rows and "A" in rows
    }
    paired_successful_comparisons = {
        f"{group}_vs_A": {
            metric: _paired_comparison(
                _successful_rows(rows[group]),
                _successful_rows(rows["A"]),
                metric,
            )
            for metric in metrics
        }
        for group in ("B", "C")
        if group in rows and "A" in rows
    }
    token_metric_valid = (
        sorted(groups) == ["A", "B", "C"]
        and all(groups[group].get("token_usage_coverage", 0.0) >= 0.90 for group in groups)
    )
    if not token_metric_valid:
        for comparison_set in (comparisons, paired_successful_comparisons):
            for comparison in comparison_set.values():
                for metric in ("input_tokens", "total_tokens"):
                    comparison[metric]["reduction_pct"] = None
                    comparison[metric]["invalid_reason"] = "token_usage_coverage_below_90pct"

    integrity = _artifact_integrity(artifact_paths, artifacts, task_ids, mode)
    experimental_validity = _experimental_validity(artifacts, task_ids)
    a_pass = groups.get("A", {}).get("pass_rate", 0.0)
    logical_reductions = {
        key: value["logical_decisions"].get("reduction_pct")
        for key, value in comparisons.items()
    }
    gates = {
        "groups_present": sorted(groups) == ["A", "B", "C"],
        "task_count": len(task_ids),
        "task_count_is_20": len(task_ids) == 20,
        "same_commit_across_groups": experimental_validity["same_commit"],
        "repetition_integrity": integrity["repetition_integrity"],
        "experimental_validity": experimental_validity["valid"],
        "coverage_B_ge_80pct": groups.get("B", {}).get("positive_chunk_coverage", 0.0) >= 0.80,
        "coverage_C_ge_80pct": groups.get("C", {}).get("positive_chunk_coverage", 0.0) >= 0.80,
        "mean_chunk_length_B_ge_1_8": groups.get("B", {}).get("positive_chunk_lengths", {}).get("mean") is not None
        and groups["B"]["positive_chunk_lengths"]["mean"] >= 1.8,
        "mean_chunk_length_C_ge_1_8": groups.get("C", {}).get("positive_chunk_lengths", {}).get("mean") is not None
        and groups["C"]["positive_chunk_lengths"]["mean"] >= 1.8,
        "correctness_B_within_5pp": groups.get("B", {}).get("pass_rate", 0.0) >= a_pass - 0.05,
        "correctness_C_within_5pp": groups.get("C", {}).get("pass_rate", 0.0) >= a_pass - 0.05,
        "logical_reduction_B_ge_15pct": (logical_reductions.get("B_vs_A") or -1.0) >= 15.0,
        "logical_reduction_C_ge_15pct": (logical_reductions.get("C_vs_A") or -1.0) >= 15.0,
        "token_metric_valid": token_metric_valid,
    }
    if (
        not gates["groups_present"]
        or not gates["task_count_is_20"]
        or not gates["same_commit_across_groups"]
        or not gates["repetition_integrity"]
        or not gates["experimental_validity"]
    ):
        conclusion = "INCONCLUSIVE"
    elif not gates["correctness_B_within_5pp"] or not gates["correctness_C_within_5pp"]:
        conclusion = "NEEDS_REVISION"
    elif not all(gates[name] for name in (
        "coverage_B_ge_80pct",
        "coverage_C_ge_80pct",
        "mean_chunk_length_B_ge_1_8",
        "mean_chunk_length_C_ge_1_8",
        "logical_reduction_B_ge_15pct",
        "logical_reduction_C_ge_15pct",
    )):
        conclusion = "INCONCLUSIVE"
    else:
        conclusion = "PASS"

    primary_group_commits = {}
    for group, group_artifacts in artifacts.items():
        commits = {
            artifact.get("runtime", {}).get("commit_sha")
            for artifact in group_artifacts
            if artifact.get("runtime", {}).get("commit_sha")
        }
        if len(commits) == 1:
            primary_group_commits[group] = next(iter(commits))
    primary_ablation = {
        "group_commits": primary_group_commits,
        "same_commit": len(primary_group_commits) == 3 and len(set(primary_group_commits.values())) == 1,
    }

    pilot_gates = {}
    pilot_status = None
    pilot_reason = ""
    if mode == "pilot":
        protocol_rates = {
            group: _protocol_error_count(rows.get(group, [])) / len(rows[group])
            if rows.get(group) else 1.0
            for group in ("A", "B", "C")
        }
        pilot_gates = {
            "task_count_is_10": len(task_ids) == 10,
            "exactly_one_repetition": integrity["exact_one_repetition"],
            "A_pass_rate_ge_80pct": groups.get("A", {}).get("pass_rate", 0.0) >= 0.80,
            "coverage_B_ge_70pct": groups.get("B", {}).get("positive_chunk_coverage", 0.0) >= 0.70,
            "coverage_C_ge_70pct": groups.get("C", {}).get("positive_chunk_coverage", 0.0) >= 0.70,
            "mean_chunk_length_B_ge_1_5": groups.get("B", {}).get("positive_chunk_lengths", {}).get("mean") is not None
            and groups["B"]["positive_chunk_lengths"]["mean"] >= 1.5,
            "mean_chunk_length_C_ge_1_5": groups.get("C", {}).get("positive_chunk_lengths", {}).get("mean") is not None
            and groups["C"]["positive_chunk_lengths"]["mean"] >= 1.5,
            "protocol_errors_not_systemic": all(rate <= 0.20 for rate in protocol_rates.values()),
        }
        pilot_status = "PASS" if all(pilot_gates.values()) else "FAIL"
        if not pilot_gates["task_count_is_10"] or not pilot_gates["exactly_one_repetition"]:
            pilot_reason = "pilot_contract_invalid"
        elif not pilot_gates["A_pass_rate_ge_80pct"]:
            pilot_reason = "insufficient_correctness"
        elif not all(pilot_gates[name] for name in (
            "coverage_B_ge_70pct",
            "coverage_C_ge_70pct",
            "mean_chunk_length_B_ge_1_5",
            "mean_chunk_length_C_ge_1_5",
        )):
            pilot_reason = "insufficient_chunk_adoption"
        elif not pilot_gates["protocol_errors_not_systemic"]:
            pilot_reason = "systemic_protocol_errors"

    return {
        "schema_version": 1,
        "mode": mode,
        "task_ids": list(task_ids),
        "groups": groups,
        "comparisons": comparisons,
        "paired_successful_comparisons": paired_successful_comparisons,
        "primary_ablation": primary_ablation,
        "artifact_integrity": integrity,
        "repetition_integrity": integrity,
        "experimental_validity": experimental_validity,
        "gates": gates,
        "pilot_gates": pilot_gates,
        "pilot_status": pilot_status,
        "pilot_reason": pilot_reason,
        "conclusion": "INCONCLUSIVE" if mode == "pilot" else conclusion,
        "final_conclusion": "INCONCLUSIVE" if mode == "pilot" else conclusion,
    }


def render_report(summary, environment):
    def format_number(number):
        return "n/a" if number is None else f"{number:.2f}"

    validity = summary.get("experimental_validity", {})

    def value(group, metric):
        if metric in {"input_tokens", "total_tokens"} and not summary.get("gates", {}).get("token_metric_valid", False):
            return "n/a"
        item = summary["groups"].get(group, {}).get(metric, {}).get("median")
        return format_number(item)

    def reduction(group, metric, comparison_set="comparisons"):
        item = summary.get(comparison_set, {}).get(f"{group}_vs_A", {}).get(metric, {})
        return format_number(item.get("reduction_pct"))

    lines = [
        "# Pico Action Chunking Real-Model Benchmark",
        "",
        f"- Conclusion: **{summary['conclusion']}**",
        f"- Commit: `{environment['commit_sha']}`",
        f"- Model: `{environment['model']}`",
        f"- Task runs: `{len(summary['task_ids'])} tasks x {environment['repetitions']} repetitions x 3 groups`",
        "",
        "## Primary A/B/C ablation",
        "",
    ]
    group_commits = summary.get("primary_ablation", {}).get("group_commits", {})
    for group in ("A", "B", "C"):
        lines.append(f"- {group} commit: `{group_commits.get(group, 'n/a')}`")
    lines.extend([
        f"- Same commit across A/B/C: `{'PASS' if validity.get('same_commit') else 'FAIL'}`",
        "- Only treatment variable: `action_chunking`",
        "",
        "## Group metrics",
        "",
        "| Group | Verifier pass | Logical decisions P25 / P50 / P75 | Provider requests P50 | Total tokens P50 | E2E ms P50 | Positive chunk coverage | Chunk length P50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group in ("A", "B", "C"):
        data = summary["groups"].get(group, {})
        logical = data.get("logical_decisions", {})
        lines.append(
            f"| {group} | {data.get('verifier_pass_rate', 0.0):.2%} | "
            f"{format_number(logical.get('p25'))} / {format_number(logical.get('median'))} / {format_number(logical.get('p75'))} | "
            f"{value(group, 'provider_requests')} | {value(group, 'total_tokens')} | {value(group, 'e2e_latency_ms')} | "
            f"{data.get('positive_chunk_coverage', 0.0):.2%} | {value(group, 'chunk_lengths')} |"
        )
    lines.extend(["", "## Paired logical-decision comparison", ""])
    for group in ("B", "C"):
        comparison = summary["comparisons"].get(f"{group}_vs_A", {}).get("logical_decisions", {})
        logical_reduction = comparison.get("reduction_pct")
        lines.append(
            f"- {group} vs A: logical-decision median mean "
            f"{format_number(comparison.get('baseline_median_mean'))} -> {format_number(comparison.get('group_median_mean'))}; "
            f"reduction `{format_number(logical_reduction)}%`" if logical_reduction is not None else f"- {group} vs A: reduction n/a"
        )
    lines.extend([
        "",
        "## All-runs efficiency comparison",
        "",
        "| Metric | A | B | Delta B/A | C | Delta C/A |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, metric in (
        ("Logical decisions / task", "logical_decisions"),
        ("Provider requests / task", "provider_requests"),
        ("Input tokens / task", "input_tokens"),
        ("Total tokens / task", "total_tokens"),
        ("E2E latency / task", "e2e_latency_ms"),
        ("Primitive submissions / task", "primitive_submissions"),
    ):
        lines.append(
            f"| {label} | {value('A', metric)} | {value('B', metric)} | {reduction('B', metric)}% | "
            f"{value('C', metric)} | {reduction('C', metric)}% |"
        )
    lines.extend([
        "",
        "## Paired-successful-runs efficiency comparison",
        "",
        "Only task IDs where both the group and A passed their verifier are included.",
    ])
    for group in ("B", "C"):
        lines.append(
            f"- {group} vs A logical decisions: "
            f"`{reduction(group, 'logical_decisions', 'paired_successful_comparisons')}%` reduction"
        )
    lines.extend(["", "## Chunk and token diagnostics", ""])
    for group in ("A", "B", "C"):
        data = summary["groups"].get(group, {})
        lines.append(
            f"- {group}: coverage `{data.get('positive_chunk_coverage', 0.0):.2%}`, "
            f"mean chunk length `{format_number(data.get('chunk_lengths', {}).get('mean'))}`, "
            f"interrupt rate `{data.get('chunk_interrupt_rate', 0.0):.2%}`, "
            f"token usage coverage `{data.get('token_usage_coverage', 0.0):.2%}`"
        )
    historical = summary.get("historical_reference") or environment.get("historical_reference")
    if historical:
        lines.extend([
            "",
            "## Historical reference",
            "",
            f"- Commit: `{historical.get('commit_sha', 'n/a')}`",
            f"- Artifact: `{historical.get('artifact', 'n/a')}`",
            "- Historical data is not used in primary Action Chunking causal comparisons.",
        ])
    compatibility = summary.get("compatibility_regression") or environment.get("compatibility_regression")
    if compatibility:
        lines.extend([
            "",
            "## Compatibility regression",
            "",
            "| Run | Commit | Passed | Chunk count zero |",
            "|---|---|---:|---:|",
        ])
        for name in ("before", "after"):
            item = compatibility.get(name, {})
            lines.append(
                f"| {name} | `{item.get('commit_sha', 'n/a')}` | "
                f"{item.get('passed', 'n/a')} | {item.get('all_chunk_count_zero', 'n/a')} |"
            )
    lines.extend(["", "## Gates", ""])
    for name, passed in summary["gates"].items():
        if name == "token_metric_valid":
            lines.append(f"- {'PASS' if passed else 'N/A'}: {name} (optional diagnostic)")
        else:
            lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend([
        "",
        "## Experimental validity",
        "",
        f"- Same Pico commit across A/B/C: `{'PASS' if validity.get('same_commit') else 'FAIL'}`",
        f"- Same model: `{'PASS' if validity.get('same_model') else 'FAIL'}`",
        f"- Same decoding config: `{'PASS' if validity.get('same_decoding') else 'FAIL'}`",
        f"- Same task IDs: `{'PASS' if validity.get('same_task_ids') else 'FAIL'}`",
        f"- Same prompt snapshot: `{'PASS' if validity.get('same_prompt_snapshot') else 'FAIL'}`",
        f"- Same fixture snapshot: `{'PASS' if validity.get('same_fixture_snapshot') else 'FAIL'}`",
        f"- Same step budget: `{'PASS' if validity.get('same_step_budget') else 'FAIL'}`",
        f"- Action Chunking is the only treatment: `{'PASS' if validity.get('action_chunking_unique_treatment') else 'FAIL'}`",
        f"- Token metric (optional): `{'PASS' if summary.get('gates', {}).get('token_metric_valid') else 'UNAVAILABLE'}`",
        f"- Chunk coverage gate: `{'PASS' if summary.get('gates', {}).get('coverage_B_ge_80pct') and summary.get('gates', {}).get('coverage_C_ge_80pct') else 'FAIL'}`",
    ])
    if summary.get("pilot_status") is not None:
        lines.extend([
            "",
            "## Pilot gate",
            "",
            f"- Status: `{summary['pilot_status']}`",
            f"- Reason: `{summary.get('pilot_reason') or 'n/a'}`",
            "- Final conclusion: `INCONCLUSIVE` (pilot is not the final experiment)",
        ])
    lines.extend([
        "",
        "## Measurement boundary",
        "",
        "- Logical decisions count model-planning rounds recorded by TaskState; provider retries are not counted as logical decisions.",
        "- E2E latency is `time.monotonic()` around `agent.ask()` and includes runtime/tool execution for the task.",
        "- Token totals are reported only when every provider attempt has input/output/total usage; missing usage remains null.",
        "- Primary A/B/C runs all execute from the current Pico commit; the historical commit is reported separately and excluded from causal comparisons.",
        "- The old 12-task regression benchmark is compatibility evidence only and is not included in this performance conclusion.",
    ])
    return "\n".join(lines) + "\n"


def _validate_contract(benchmark):
    tasks = benchmark["tasks"]
    expected = [f"MF-{index:02d}" for index in range(1, 7)]
    expected += [f"MS-{index:02d}" for index in range(1, 5)]
    expected += [f"CT-{index:02d}" for index in range(1, 5)]
    expected += [f"OB-{index:02d}" for index in range(1, 4)]
    expected += [f"SC-{index:02d}" for index in range(1, 4)]
    actual = [task["id"] for task in tasks]
    if actual != expected:
        raise ValueError("benchmark task IDs/order do not match the fixed contract")
    if any(task["allowed_tools"] != ALLOWED_TOOLS for task in tasks):
        raise ValueError("all benchmark tasks must use exactly the fixed read-only tool set")
    if any(int(task["step_budget"]) != 8 for task in tasks):
        raise ValueError("all benchmark tasks must use step_budget=8")
    if any(task["fixture_repo"] != "benchmarks/fixtures/action_chunk_repo" for task in tasks):
        raise ValueError("all benchmark tasks must use the fixed action chunk fixture")
    if any(
        Path(str(task.get("artifact_path", ""))).is_absolute()
        or ".." in Path(str(task.get("artifact_path", ""))).parts
        for task in tasks
    ):
        raise ValueError("benchmark artifact paths must stay inside the fixture")
    if any(not str(task["verifier"]).lstrip().startswith("python -c ") for task in tasks):
        raise ValueError("benchmark verifiers must use the fixed Python -c form")
    if any("check_semantic" not in task["verifier"] for task in tasks):
        raise ValueError("all action chunking tasks must use the semantic verifier")


def _preflight(api_key, base_url, timeout, client_factory=None):
    def blocked_phase(reason):
        return {"status": "blocked", "reason": reason}

    if not api_key:
        reason = "PICO_OPENAI_API_KEY is not set"
        return {
            "status": "blocked",
            "reason": reason,
            "provider": blocked_phase(reason),
            "pico_tool_smoke": blocked_phase("provider preflight did not pass"),
            "pico_chunk_smoke": blocked_phase("provider preflight did not pass"),
        }

    if client_factory is None:
        client_factory = lambda: OpenAICompatibleModelClient(
            model=MODEL_NAME,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            timeout=timeout,
        )

    client = client_factory()
    started_at = time.monotonic()
    try:
        response = client.complete(
            "Reply with exactly PICO_RESPONSES_PREFLIGHT_OK and no other text.",
            16,
            logical_decision_id="preflight",
        )
    except Exception as exc:  # noqa: BLE001 - preflight must turn any provider failure into evidence
        return {
            "status": "failed",
            "reason": _safe_error(exc, api_key),
            "elapsed_ms": int((time.monotonic() - started_at) * 1000),
            "provider_requests": len(getattr(client, "last_provider_attempts", []) or []),
        }
    usage = aggregate_provider_usage(getattr(client, "last_provider_attempts", []) or [])
    response_match = str(response).strip() == "PICO_RESPONSES_PREFLIGHT_OK"
    provider = {
        "status": "passed" if response_match else "failed",
        "response_match": response_match,
        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
        "provider_requests": len(getattr(client, "last_provider_attempts", []) or []),
        "token_usage_coverage": usage["token_usage_coverage"],
        "token_usage_complete": usage["token_usage_complete"],
        "response_text_recorded": False,
    }
    if provider["status"] != "passed":
        reason = "provider response did not match the preflight contract"
        return {
            "status": "failed",
            "reason": reason,
            "provider": provider,
            "pico_tool_smoke": blocked_phase(reason),
            "pico_chunk_smoke": blocked_phase(reason),
        }

    tool_smoke = _run_pico_smoke(
        client_factory(),
        action_chunking={"enabled": False},
        allowed_tools=["read_file"],
        prompt="Read app/config.py and report the actual configured timeout value.",
    )
    if tool_smoke["status"] == "passed" and tool_smoke["primitive_submissions"] >= 1:
        tool_smoke["status"] = "passed"
    else:
        tool_smoke["status"] = "blocked"
        tool_smoke["reason"] = tool_smoke.get("reason") or "model did not complete Pico tool protocol"

    chunk_smoke = _run_pico_smoke(
        client_factory(),
        action_chunking=GROUP_CONFIGS["B"],
        allowed_tools=ALLOWED_TOOLS,
        prompt=(
            "Inspect app/config.py, app/parser.py, and app/loader.py. "
            "They are independent and all file paths are already known."
        ),
    )
    if chunk_smoke["status"] == "passed" and chunk_smoke["chunk_count"] > 0 and chunk_smoke["mean_chunk_length"] >= 2:
        chunk_smoke["status"] = "passed"
    else:
        chunk_smoke["status"] = "blocked"
        chunk_smoke["reason"] = "model_did_not_adopt_chunk_protocol"

    all_passed = tool_smoke["status"] == "passed" and chunk_smoke["status"] == "passed"
    return {
        "status": "passed" if all_passed else "blocked",
        "reason": "" if all_passed else (chunk_smoke.get("reason") or tool_smoke.get("reason")),
        "provider": provider,
        "pico_tool_smoke": tool_smoke,
        "pico_chunk_smoke": chunk_smoke,
    }


def run_real_benchmark(
    benchmark_path,
    groups,
    repetitions,
    temperature,
    max_new_tokens,
    timeout,
    output_dir,
    task_ids=None,
    api_key=None,
    base_url=None,
    mode="final",
):
    benchmark_path = Path(benchmark_path).resolve()
    benchmark = load_benchmark(benchmark_path)
    _validate_contract(benchmark)
    task_ids = list(task_ids or [task["id"] for task in benchmark["tasks"]])
    unknown = sorted(set(task_ids) - {task["id"] for task in benchmark["tasks"]})
    if unknown:
        raise ValueError("unknown task IDs: " + ", ".join(unknown))
    groups = list(groups)
    if len(groups) != 3 or any(group not in GROUP_CONFIGS for group in groups) or set(groups) != {"A", "B", "C"}:
        raise ValueError("real benchmark requires exactly groups A B C")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must not repeat")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    base_url = base_url or os.environ.get("PICO_OPENAI_API_BASE", DEFAULT_BASE_URL)
    api_key = api_key if api_key is not None else os.environ.get("PICO_OPENAI_API_KEY", "")
    model_from_env = os.environ.get("PICO_OPENAI_MODEL", "")
    if model_from_env and model_from_env != MODEL_NAME:
        raise ValueError(f"PICO_OPENAI_MODEL must be {MODEL_NAME}")

    commit_sha = _git_value(["rev-parse", "HEAD"], cwd=REPO_ROOT)
    timestamp = datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).strftime("%Y%m%d-%H%M%S%f")
    run_root = Path(output_dir).resolve() / f"{timestamp}-{commit_sha[:12]}"
    run_root.mkdir(parents=True, exist_ok=False)
    fixture_paths = [REPO_ROOT / str(task["fixture_repo"]) for task in benchmark["tasks"] if task["id"] in task_ids]
    environment = {
        "schema_version": 1,
        "mode": mode,
        "commit_sha": commit_sha,
        "model": MODEL_NAME,
        "base_url": base_url,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "timeout": timeout,
        "approval": "auto",
        "task_ids": task_ids,
        "task_prompt_snapshot_id": _task_prompt_snapshot_id(benchmark["tasks"], task_ids),
        "step_budget": 8,
        "repetitions": repetitions,
        "group_order_by_repetition": {
            str(index): list(ROTATIONS[(index - 1) % len(ROTATIONS)])
            for index in range(1, repetitions + 1)
        },
        "groups": groups,
        "group_configs": {group: normalize_action_chunking(GROUP_CONFIGS[group]) for group in groups},
        "fixture_snapshot_id": _fixture_snapshot_id(fixture_paths),
        "timezone": DEFAULT_TIMEZONE,
        "locale": _current_locale(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "api_key_present": bool(api_key),
        "api_key_source": "PICO_OPENAI_API_KEY",
        "baseline_commit_sha": ORIGINAL_BASELINE_SHA,
        "historical_baseline_commit_sha": ORIGINAL_BASELINE_SHA,
        "group_commits": {group: commit_sha for group in groups},
    }
    preflight = _preflight(api_key, base_url, timeout)
    environment["preflight"] = preflight
    _json_write(run_root / "environment.json", environment)
    if preflight.get("status") != "passed":
        experimental_validity = {
            "same_commit": False,
            "same_model": False,
            "same_decoding": False,
            "same_task_ids": False,
            "same_prompt_snapshot": False,
            "same_fixture_snapshot": False,
            "same_step_budget": False,
            "same_execution_config": False,
            "action_chunking_unique_treatment": False,
            "all_required_fields_consistent": False,
            "valid": False,
        }
        summary = {
            "schema_version": 1,
            "mode": mode,
            "task_ids": task_ids,
            "groups": {},
            "comparisons": {},
            "gates": {
                "groups_present": False,
                "task_count": len(task_ids),
                "task_count_is_20": len(task_ids) == 20,
                "same_commit_across_groups": False,
                "repetition_integrity": False,
                "experimental_validity": False,
                "token_metric_valid": False,
            },
            "primary_ablation": {
                "group_commits": {},
                "same_commit": False,
            },
            "artifact_integrity": {},
            "repetition_integrity": {},
            "experimental_validity": experimental_validity,
            "pilot_gates": {},
            "pilot_status": "BLOCKED" if mode == "pilot" else None,
            "pilot_reason": "preflight_not_passed" if mode == "pilot" else "",
            "conclusion": "INCONCLUSIVE",
            "final_conclusion": "INCONCLUSIVE",
            "preflight": preflight,
        }
        _json_write(run_root / "summary.json", summary)
        (run_root / "report.md").write_text(
            "# Pico Action Chunking Real-Model Benchmark\n\n"
            f"- Conclusion: **INCONCLUSIVE**\n- Preflight: `{preflight.get('status')}`\n"
            f"- Reason: {preflight.get('reason', 'provider preflight did not pass')}\n",
            encoding="utf-8",
        )
        return run_root, summary

    historical_reference = {
        "status": "not_run",
        "commit_sha": ORIGINAL_BASELINE_SHA,
        "not_used_in_primary_comparisons": True,
    }
    compatibility_regression = {"status": "not_run"}
    try:
        with _historical_baseline_worktree_context(benchmark_path) as baseline_worktree:
            environment["baseline_setup"] = {
                "status": "passed",
                "worktree": str(baseline_worktree),
                "commit_sha": ORIGINAL_BASELINE_SHA,
            }
            try:
                historical_reference = _run_historical_reference(
                    baseline_worktree,
                    run_root,
                    task_ids,
                    api_key,
                    base_url,
                    temperature,
                    max_new_tokens,
                    timeout,
                )
            except Exception as exc:  # noqa: BLE001 - keep primary experiment independent
                historical_reference = {
                    "status": "failed",
                    "commit_sha": ORIGINAL_BASELINE_SHA,
                    "reason": _safe_error(exc, api_key),
                    "not_used_in_primary_comparisons": True,
                }
            try:
                compatibility_regression = _run_compatibility_regression(baseline_worktree, run_root)
                compatibility_regression["status"] = "passed" if all(
                    compatibility_regression.get(name, {}).get("total_tasks") == 12
                    and compatibility_regression.get(name, {}).get("passed") == 12
                    and compatibility_regression.get(name, {}).get("all_chunk_count_zero")
                    for name in ("before", "after")
                ) else "failed"
            except Exception as exc:  # noqa: BLE001 - preserve real-run evidence if compatibility setup fails
                compatibility_regression = {
                    "status": "failed",
                    "reason": _safe_error(exc, api_key),
                }
    except Exception as exc:  # noqa: BLE001 - historical evidence must not contaminate primary A/B/C
        environment["baseline_setup"] = {"status": "failed", "reason": _safe_error(exc, api_key)}
        historical_reference = {
            "status": "failed",
            "commit_sha": ORIGINAL_BASELINE_SHA,
            "reason": _safe_error(exc, api_key),
            "not_used_in_primary_comparisons": True,
        }
        compatibility_regression = {
            "status": "failed",
            "reason": _safe_error(exc, api_key),
        }
    environment["compatibility_regression"] = compatibility_regression
    environment["historical_reference"] = historical_reference
    _json_write(run_root / "environment.json", environment)

    artifact_paths = {group: [] for group in groups}
    for repetition in range(1, repetitions + 1):
        order = ROTATIONS[(repetition - 1) % len(ROTATIONS)]
        for group in order:
            if group not in groups:
                continue
            artifact_path = run_root / group / f"rep-{repetition:02d}.json"
            evaluator = BenchmarkEvaluator(
                benchmark_path=benchmark_path,
                artifact_path=artifact_path,
                workspace_root=run_root / "workspaces" / group / f"rep-{repetition:02d}",
                model_name=MODEL_NAME,
                model_version="openai-compatible-responses",
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                timezone_name=DEFAULT_TIMEZONE,
                model_client_factory=lambda task, workspace: OpenAICompatibleModelClient(
                    model=MODEL_NAME,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=timeout,
                ),
                action_chunking=GROUP_CONFIGS[group],
            )
            artifact = evaluator.run(task_ids=task_ids, continue_on_error=True)
            _annotate_primary_artifact(artifact, group, repetition, task_ids, environment)
            _json_write(artifact_path, _redact(artifact, api_key))
            artifact_paths[group].append(artifact_path)

    summary = summarize_real_artifacts(artifact_paths, task_ids, mode=mode)
    summary["preflight"] = preflight
    summary["compatibility_regression"] = compatibility_regression
    summary["historical_reference"] = historical_reference
    if mode != "pilot" and compatibility_regression.get("status") != "passed" and summary["conclusion"] == "PASS":
        summary["conclusion"] = "NEEDS_REVISION"
        summary["final_conclusion"] = "NEEDS_REVISION"
    _json_write(run_root / "summary.json", summary)
    (run_root / "report.md").write_text(render_report(summary, environment), encoding="utf-8")
    return run_root, summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmarks/action_chunking_tasks.json")
    parser.add_argument("--groups", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pilot", action="store_true", help="Run the fixed 10-task, one-repetition pilot.")
    parser.add_argument("--tasks", default="", help="Comma-separated fixed task IDs; use only for a documented subset/pilot.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    task_ids = [item.strip() for item in args.tasks.split(",") if item.strip()] if args.tasks else None
    if args.pilot:
        task_ids = PILOT_TASK_IDS
        repetitions = 1
    else:
        repetitions = args.repetitions
    base_url = os.environ.get("PICO_OPENAI_API_BASE", DEFAULT_BASE_URL)
    api_key = os.environ.get("PICO_OPENAI_API_KEY", "")
    if args.preflight_only:
        result = _preflight(api_key, base_url, args.timeout)
        print(json.dumps({key: value for key, value in result.items() if key != "response_text"}, sort_keys=True))
        return 0 if result.get("status") == "passed" else 2
    run_root, summary = run_real_benchmark(
        benchmark_path=args.benchmark,
        groups=args.groups,
        repetitions=repetitions,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        output_dir=args.output_dir,
        task_ids=task_ids,
        api_key=api_key,
        base_url=base_url,
        mode="pilot" if args.pilot else "final",
    )
    print(f"artifact_root={run_root}")
    if args.pilot:
        print(f"pilot_status={summary.get('pilot_status', 'BLOCKED')}")
        return 0 if summary.get("pilot_status") == "PASS" else 2
    print(f"conclusion={summary.get('final_conclusion', summary.get('conclusion', 'INCONCLUSIVE'))}")
    return 0 if summary.get("final_conclusion", summary.get("conclusion")) == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
