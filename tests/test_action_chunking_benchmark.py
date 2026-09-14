import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pico.evaluation.evaluator import BenchmarkEvaluator, load_benchmark
from scripts.run_action_chunking_real_benchmark import (
    GROUP_CONFIGS,
    POSITIVE_CATEGORIES,
    _aggregate_group,
    run_deterministic_benchmark,
    run_real_benchmark,
    summarize_real_artifacts,
)


def test_real_benchmark_contract_is_fixed_to_twenty_read_only_tasks():
    benchmark = load_benchmark("benchmarks/action_chunking_tasks.json")
    assert [task["id"] for task in benchmark["tasks"]] == (
        [f"MF-{index:02d}" for index in range(1, 7)]
        + [f"MS-{index:02d}" for index in range(1, 5)]
        + [f"CT-{index:02d}" for index in range(1, 5)]
        + [f"OB-{index:02d}" for index in range(1, 4)]
        + [f"SC-{index:02d}" for index in range(1, 4)]
    )
    assert len(benchmark["tasks"]) == 20
    assert {task["category"] for task in benchmark["tasks"]} == {
        "known_path_multi_file_inspection",
        "multi_symbol_search",
        "config_impl_test_inspection",
        "observation_boundary",
        "sequential_control",
    }
    for task in benchmark["tasks"]:
        assert task["allowed_tools"] == ["list_files", "read_file", "search"]
        assert task["step_budget"] == 8


def test_fake_model_a_b_c_contract_has_chunk_coverage_and_boundary_interrupts(tmp_path):
    results = run_deterministic_benchmark(
        benchmark_path="benchmarks/action_chunking_tasks.json",
        workspace_root=tmp_path,
    )

    assert set(results) == set(GROUP_CONFIGS)
    for artifact in results.values():
        assert artifact["summary"]["total_tasks"] == 20
        assert artifact["summary"]["passed"] == 20
        assert artifact["summary"]["verifier_pass_rate"] == 1.0
        assert all(row["input_tokens"] is None for row in artifact["rows"])

    for group in ("B", "C"):
        positive_rows = [row for row in results[group]["rows"] if row["category"] in POSITIVE_CATEGORIES]
        assert len(positive_rows) == 14
        assert all(row["chunk_count"] > 0 for row in positive_rows)
        aggregate = _aggregate_group(results[group]["rows"])
        assert aggregate["positive_chunk_coverage"] == 1.0
        assert aggregate["chunk_lengths"]["median"] >= 1.8

    assert any(
        row["task_state"]["chunk_interrupts"] > 0
        for row in results["C"]["rows"]
        if row["category"] == "observation_boundary"
    )
    assert all(
        row["chunk_count"] == 0
        for row in results["B"]["rows"] + results["C"]["rows"]
        if row["category"] == "sequential_control"
    )

    task_ids = [row["id"] for row in results["A"]["rows"]]
    summary = summarize_real_artifacts(
        {group: [tmp_path / f"deterministic-{group}.json"] for group in GROUP_CONFIGS},
        task_ids,
    )
    assert summary["conclusion"] == "PASS"
    assert summary["comparisons"]["B_vs_A"]["logical_decisions"]["reduction_pct"] > 15


def test_real_runner_records_blocked_preflight_without_running_tasks(tmp_path):
    artifact_root, summary = run_real_benchmark(
        benchmark_path="benchmarks/action_chunking_tasks.json",
        groups=["A", "B", "C"],
        repetitions=1,
        temperature=0.0,
        max_new_tokens=768,
        timeout=300,
        output_dir=tmp_path,
        api_key="",
        base_url="https://api.example.invalid/v1",
        task_ids=["MF-01"],
    )

    assert summary["conclusion"] == "INCONCLUSIVE"
    assert summary["preflight"]["status"] == "blocked"
    assert (artifact_root / "environment.json").exists()
    assert not list(artifact_root.glob("A/rep-*.json"))


def test_real_group_records_provider_failure_without_dropping_fixed_tasks(tmp_path):
    def failing_factory(task, workspace):
        del task, workspace
        raise RuntimeError("provider unavailable")

    artifact = BenchmarkEvaluator(
        benchmark_path="benchmarks/action_chunking_tasks.json",
        artifact_path=tmp_path / "group.json",
        workspace_root=tmp_path / "workspaces",
        model_client_factory=failing_factory,
    ).run(task_ids=["MF-01", "MF-02"], continue_on_error=True)

    assert [row["id"] for row in artifact["rows"]] == ["MF-01", "MF-02"]
    assert all(row["status"] == "fail" for row in artifact["rows"])
    assert artifact["summary"]["total_tasks"] == 2
