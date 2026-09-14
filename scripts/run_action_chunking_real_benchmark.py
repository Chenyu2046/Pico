"""Run the fixed A/B/C Action Chunking benchmark.

The runner keeps credentials in the process environment and writes only
non-secret experiment metadata.  Real-model results are meaningful only when
the provider preflight succeeds and the final gates pass.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
    load_benchmark,
    run_fixed_benchmark,
)
from pico.providers.clients import FakeModelClient, OpenAICompatibleModelClient

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


def _prepare_baseline_worktree(benchmark_path):
    worktree = Path(tempfile.mkdtemp(prefix="pico-action-chunk-baseline-"))
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), ORIGINAL_BASELINE_SHA],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("could not create original baseline worktree: " + _safe_error(result.stderr))
    benchmark_target = worktree / "benchmarks" / "action_chunking_tasks.json"
    benchmark_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(benchmark_path, benchmark_target)
    fixture_target = worktree / "benchmarks" / "fixtures" / "action_chunk_repo"
    fixture_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "benchmarks" / "fixtures" / "action_chunk_repo", fixture_target)
    helper_target = worktree / "scripts" / "run_action_chunking_baseline.py"
    helper_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "run_action_chunking_baseline.py", helper_target)
    return worktree


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
    result = subprocess.run(command, cwd=worktree, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("original baseline runner failed: " + _safe_error(result.stderr, api_key))
    return json.loads(Path(artifact_path).read_text(encoding="utf-8"))


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
    return [
        _tool(
            "search",
            {"pattern": pattern, "path": "tests" if pattern.startswith("test_") else "app"},
        )
        for pattern in workload.get("expected_patterns", [])
    ]


def _deterministic_outputs(task, action_chunking):
    actions = _planned_actions(task)
    final = "<final>Inspection complete. " + task["prompt"] + "</final>"
    if not action_chunking.get("enabled"):
        return actions + [final]

    if task.get("category") == "observation_boundary":
        return [_chunk([actions[0]])] + [_chunk(actions[1:]), final]

    if task.get("category") in POSITIVE_CATEGORIES:
        chunks = [_chunk(actions[index:index + 4]) for index in range(0, len(actions), 4)]
        return chunks + [final]

    return actions + [final]


def deterministic_model_factory(action_chunking):
    def factory(task, workspace):
        del workspace
        return FakeModelClient(_deterministic_outputs(task, action_chunking))

    return factory


def run_deterministic_benchmark(
    benchmark_path="benchmarks/action_chunking_tasks.json",
    workspace_root=None,
):
    """Run all 20 tasks through FakeModelClient for contract validation."""
    benchmark_path = Path(benchmark_path).resolve()
    if workspace_root is None:
        workspace_root = tempfile.mkdtemp(prefix="pico-action-chunk-deterministic-")
    result = {}
    for group, config in GROUP_CONFIGS.items():
        evaluator = BenchmarkEvaluator(
            benchmark_path=benchmark_path,
            artifact_path=Path(workspace_root) / f"deterministic-{group}.json",
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
        result[group] = evaluator.run()
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
        "chunk_count": _distribution(_numeric_values(rows, "chunk_count")),
        "token_usage_coverage": sum(float(row.get("token_usage_coverage", 0.0)) for row in rows) / len(rows) if rows else 0.0,
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


def summarize_real_artifacts(artifact_paths, task_ids):
    artifacts = {group: [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths] for group, paths in artifact_paths.items()}
    rows = {group: [row for artifact in group_artifacts for row in artifact.get("rows", [])] for group, group_artifacts in artifacts.items()}
    groups = {group: _aggregate_group(group_rows) for group, group_rows in rows.items()}
    comparisons = {
        f"{group}_vs_A": {
            metric: _paired_comparison(rows[group], rows["A"], metric)
            for metric in ("logical_decisions", "provider_requests", "input_tokens", "total_tokens", "e2e_latency_ms")
        }
        for group in ("B", "C")
        if group in rows
    }
    paired_successful_comparisons = {
        f"{group}_vs_A": {
            metric: _paired_comparison(
                _successful_rows(rows[group]),
                _successful_rows(rows["A"]),
                metric,
            )
            for metric in ("logical_decisions", "provider_requests", "input_tokens", "total_tokens", "e2e_latency_ms")
        }
        for group in ("B", "C")
        if group in rows
    }
    a_pass = groups.get("A", {}).get("pass_rate", 0.0)
    logical_reductions = {
        key: value["logical_decisions"].get("reduction_pct")
        for key, value in comparisons.items()
    }
    gates = {
        "groups_present": sorted(groups) == ["A", "B", "C"],
        "task_count": len(task_ids),
        "task_count_is_20": len(task_ids) == 20,
        "coverage_B_ge_80pct": groups.get("B", {}).get("positive_chunk_coverage", 0.0) >= 0.80,
        "coverage_C_ge_80pct": groups.get("C", {}).get("positive_chunk_coverage", 0.0) >= 0.80,
        "mean_chunk_length_B_ge_1_8": groups.get("B", {}).get("chunk_lengths", {}).get("mean") is not None and groups["B"]["chunk_lengths"]["mean"] >= 1.8,
        "mean_chunk_length_C_ge_1_8": groups.get("C", {}).get("chunk_lengths", {}).get("mean") is not None and groups["C"]["chunk_lengths"]["mean"] >= 1.8,
        "correctness_B_within_5pp": groups.get("B", {}).get("pass_rate", 0.0) >= a_pass - 0.05,
        "correctness_C_within_5pp": groups.get("C", {}).get("pass_rate", 0.0) >= a_pass - 0.05,
        "logical_reduction_B_ge_15pct": (logical_reductions.get("B_vs_A") or -1.0) >= 15.0,
        "logical_reduction_C_ge_15pct": (logical_reductions.get("C_vs_A") or -1.0) >= 15.0,
    }
    if not gates["groups_present"] or not gates["task_count_is_20"]:
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
    return {
        "schema_version": 1,
        "task_ids": list(task_ids),
        "groups": groups,
        "comparisons": comparisons,
        "paired_successful_comparisons": paired_successful_comparisons,
        "gates": gates,
        "conclusion": conclusion,
    }


def render_report(summary, environment):
    def format_number(number):
        return "n/a" if number is None else f"{number:.2f}"

    def value(group, metric):
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
        "## Group metrics",
        "",
        "| Group | Verifier pass | Logical decisions P25 / P50 / P75 | Provider requests P50 | Total tokens P50 | E2E ms P50 | Positive chunk coverage | Chunk length P50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
    compatibility = summary.get("compatibility_regression") or environment.get("compatibility_regression")
    if compatibility:
        lines.extend([
            "",
            "## Existing 12-task regression",
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
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend([
        "",
        "## Measurement boundary",
        "",
        "- Logical decisions count model-planning rounds recorded by TaskState; provider retries are not counted as logical decisions.",
        "- E2E latency is `time.monotonic()` around `agent.ask()` and includes runtime/tool execution for the task.",
        "- Token totals are reported only when every provider attempt has input/output/total usage; missing usage remains null.",
        "- Group A runs from the original baseline commit while groups B/C run from the current commit; this preserves a true ReAct baseline but includes the commit delta in the comparison.",
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


def _preflight(api_key, base_url, timeout):
    if not api_key:
        return {"status": "blocked", "reason": "PICO_OPENAI_API_KEY is not set"}
    client = OpenAICompatibleModelClient(
        model=MODEL_NAME,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        timeout=timeout,
    )
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
    return {
        "status": "passed" if str(response).strip() == "PICO_RESPONSES_PREFLIGHT_OK" else "failed",
        "response_match": str(response).strip() == "PICO_RESPONSES_PREFLIGHT_OK",
        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
        "provider_requests": len(getattr(client, "last_provider_attempts", []) or []),
        "response_text_recorded": False,
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
):
    benchmark_path = Path(benchmark_path)
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
    run_root = Path(output_dir) / f"{timestamp}-{commit_sha[:12]}"
    run_root.mkdir(parents=True, exist_ok=False)
    fixture_paths = [REPO_ROOT / str(task["fixture_repo"]) for task in benchmark["tasks"] if task["id"] in task_ids]
    environment = {
        "schema_version": 1,
        "commit_sha": commit_sha,
        "model": MODEL_NAME,
        "base_url": base_url,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "timeout": timeout,
        "approval": "auto",
        "task_ids": task_ids,
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
        "group_commits": {"A": ORIGINAL_BASELINE_SHA, "B": commit_sha, "C": commit_sha},
    }
    preflight = _preflight(api_key, base_url, timeout)
    environment["preflight"] = preflight
    _json_write(run_root / "environment.json", environment)
    if preflight.get("status") != "passed":
        summary = {"schema_version": 1, "task_ids": task_ids, "groups": {}, "comparisons": {}, "gates": {}, "conclusion": "INCONCLUSIVE", "preflight": preflight}
        _json_write(run_root / "summary.json", summary)
        (run_root / "report.md").write_text(
            "# Pico Action Chunking Real-Model Benchmark\n\n"
            f"- Conclusion: **INCONCLUSIVE**\n- Preflight: `{preflight.get('status')}`\n"
            f"- Reason: {preflight.get('reason', 'provider preflight did not pass')}\n",
            encoding="utf-8",
        )
        return run_root, summary

    try:
        baseline_worktree = _prepare_baseline_worktree(benchmark_path)
    except Exception as exc:
        environment["baseline_setup"] = {"status": "failed", "reason": _safe_error(exc, api_key)}
        _json_write(run_root / "environment.json", environment)
        raise
    environment["baseline_setup"] = {
        "status": "passed",
        "worktree": str(baseline_worktree),
        "commit_sha": ORIGINAL_BASELINE_SHA,
    }
    try:
        compatibility_regression = _run_compatibility_regression(baseline_worktree, run_root)
        compatibility_regression["status"] = "passed" if all(
            item["total_tasks"] == 12 and item["passed"] == 12 and item["all_chunk_count_zero"]
            for item in compatibility_regression.values()
        ) else "failed"
    except Exception as exc:  # noqa: BLE001 - preserve real-run evidence if compatibility setup fails
        compatibility_regression = {
            "status": "failed",
            "reason": _safe_error(exc, api_key),
        }
    environment["compatibility_regression"] = compatibility_regression
    _json_write(run_root / "environment.json", environment)

    artifact_paths = {group: [] for group in groups}
    for repetition in range(1, repetitions + 1):
        order = ROTATIONS[(repetition - 1) % len(ROTATIONS)]
        for group in order:
            if group not in groups:
                continue
            artifact_path = run_root / group / f"rep-{repetition:02d}.json"
            if group == "A":
                _run_baseline_task_set(
                    baseline_worktree,
                    artifact_path,
                    run_root / "workspaces" / group / f"rep-{repetition:02d}",
                    task_ids,
                    api_key,
                    base_url,
                    temperature,
                    max_new_tokens,
                    timeout,
                )
                artifact_paths[group].append(artifact_path)
                continue
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
            _json_write(artifact_path, _redact(artifact, api_key))
            artifact_paths[group].append(artifact_path)

    summary = summarize_real_artifacts(artifact_paths, task_ids)
    summary["preflight"] = preflight
    summary["compatibility_regression"] = compatibility_regression
    if compatibility_regression.get("status") != "passed" and summary["conclusion"] == "PASS":
        summary["conclusion"] = "NEEDS_REVISION"
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
    )
    print(f"artifact_root={run_root}")
    print(f"conclusion={summary.get('conclusion', 'INCONCLUSIVE')}")
    return 0 if summary.get("conclusion") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
