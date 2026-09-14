import hashlib
import json
import locale as locale_module
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..action_chunk import normalize_action_chunking
from ..features import memory as memorylib
from ..providers.clients import FakeModelClient
from ..run_store import RunStore
from ..runtime import Pico, SessionStore
from ..task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from ..tools import legal_tool_names
from ..workspace import WorkspaceContext

BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_PATH = Path("benchmarks/coding_tasks.json")
DEFAULT_ARTIFACT_PATH = Path("benchmarks/benchmark-v1.json")
DEFAULT_HARNESS_REGRESSION_V2_ARTIFACT_PATH = Path("artifacts/harness-regression-v2.json")
DEFAULT_MODEL_NAME = "FakeModelClient"
DEFAULT_MODEL_VERSION = "scripted-deterministic"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TIMEZONE = "Asia/Shanghai"

ACTION_CHUNKING_BENCHMARK_SCHEMA_VERSION = 1
_ACTION_CHUNKING_ALLOWED_TOOLS = list(("list_files", "read_file", "search"))
ACTION_CHUNKING_BENCHMARK_MATRIX = (
    {
        "id": "A",
        "name": "react_single_action",
        "semantics": {
            "chunking": "disabled",
            "boundary": "single primitive action per logical decision",
            "skill_guidance": False,
        },
        "action_chunking": normalize_action_chunking({"enabled": False}),
    },
    {
        "id": "B",
        "name": "chunk_hard_boundaries",
        "semantics": {
            "chunking": "variable length read-only chunks",
            "boundary": "runtime hard boundaries only",
            "skill_guidance": False,
        },
        "action_chunking": normalize_action_chunking(
            {
                "enabled": True,
                "allowed_tools": _ACTION_CHUNKING_ALLOWED_TOOLS,
                # Keep B's observation boundary out of the fixed workload;
                # C makes this boundary an explicit treatment variable.
                "observation_budget_chars": 1_000_000,
            }
        ),
    },
    {
        "id": "C",
        "name": "chunk_observation_boundary",
        "semantics": {
            "chunking": "variable length read-only chunks",
            "boundary": "hard boundaries plus observation budget/replanning",
            "skill_guidance": False,
        },
        "action_chunking": normalize_action_chunking(
            {
                "enabled": True,
                "allowed_tools": _ACTION_CHUNKING_ALLOWED_TOOLS,
                "observation_budget_chars": 12_000,
            }
        ),
    },
    {
        "id": "D",
        "name": "chunk_observation_and_skill_guidance",
        "semantics": {
            "chunking": "variable length read-only chunks",
            "boundary": "hard boundaries plus observation budget/replanning",
            "skill_guidance": True,
        },
        "action_chunking": normalize_action_chunking(
            {
                "enabled": True,
                "allowed_tools": _ACTION_CHUNKING_ALLOWED_TOOLS,
                "observation_budget_chars": 12_000,
                "skill_guidance_enabled": True,
            }
        ),
    },
)


def build_action_chunking_benchmark_matrix():
    """Return the fixed A/B/C/D treatment matrix without running it."""
    return deepcopy(list(ACTION_CHUNKING_BENCHMARK_MATRIX))

REQUIRED_BENCHMARK_KEYS = ("schema_version", "tasks")
REQUIRED_TASK_KEYS = (
    "id",
    "prompt",
    "fixture_repo",
    "allowed_tools",
    "step_budget",
    "expected_artifact",
    "verifier",
    "category",
)

TASK_FIXTURE_ARTIFACTS = {
    "bench_repo_readme": "README.md",
    "bench_repo_patch": "sample.txt",
}

SCRIPTED_MODEL_OUTPUTS = {
    "readme_intro_locked": [
        '<tool name="patch_file" path="README.md"><old_text>This is a placeholder benchmark fixture.</old_text><new_text>This fixture is a locked benchmark workspace.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "readme_schema_note": [
        '<tool name="patch_file" path="README.md"><old_text>- Placeholder note about the repo.</old_text><new_text>- The benchmark schema and baseline are fixed.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "readme_ordering_note": [
        '<tool name="patch_file" path="README.md"><old_text>- Placeholder note about the file layout.</old_text><new_text>- Deterministic file ordering keeps benchmark diffs stable.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "sample_beta_locked": [
        '<tool name="patch_file" path="sample.txt"><old_text>beta</old_text><new_text>beta-locked</new_text></tool>',
        "<final>Done.</final>",
    ],
    "sample_gamma_locked": [
        '<tool name="patch_file" path="sample.txt"><old_text>gamma</old_text><new_text>gamma-locked</new_text></tool>',
        "<final>Done.</final>",
    ],
    "sample_placeholder_delta": [
        '<tool name="patch_file" path="sample.txt"><old_text>placeholder</old_text><new_text>delta</new_text></tool>',
        "<final>Done.</final>",
    ],
    "invalid_patch_recovery": [
        '<tool>{"name":"patch_file","args":{"path":"README.md","old_text":"This is a placeholder benchmark fixture."}}</tool>',
        '<tool name="patch_file" path="README.md"><old_text>This is a placeholder benchmark fixture.</old_text><new_text>This fixture recovered after invalid patch args.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "path_escape_recovery": [
        '<tool>{"name":"read_file","args":{"path":"../outside.txt","start":1,"end":1}}</tool>',
        '<tool name="patch_file" path="sample.txt"><old_text>alpha</old_text><new_text>alpha-guarded</new_text></tool>',
        "<final>Done.</final>",
    ],
    "repeated_read_recovery": [
        '<tool>{"name":"read_file","args":{"path":"sample.txt","start":1,"end":4}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"sample.txt","start":1,"end":4}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"sample.txt","start":1,"end":4}}</tool>',
        '<tool name="patch_file" path="sample.txt"><old_text>placeholder</old_text><new_text>repeat-guarded</new_text></tool>',
        "<final>Done.</final>",
    ],
    "context_reduction_checkpoint": [
        "<final>Done.</final>",
    ],
    "freshness_reanchor_resume": [
        "<final>Done.</final>",
    ],
    "workspace_mismatch_resume": [
        "<final>Done.</final>",
    ],
    "durable_promotion_accept": [
        "<final>Project convention: Preserve benchmark regression artifacts under artifacts/.\nDecision: Keep harness regression deterministic and reproducible.</final>",
    ],
    "durable_promotion_reject": [
        "<final>Project convention: Keep verifier outcomes stable across reruns.\nDependency: API key is sk-benchmark-secret.\nDecision: Current goal is debug the harness.</final>",
    ],
}


def _git_value(args, fallback="", cwd=None):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or fallback
    except Exception:
        return fallback


def _current_locale():
    try:
        return locale_module.setlocale(locale_module.LC_CTYPE)
    except Exception:
        return locale_module.getdefaultlocale()[0] or "C"


def _now_in_timezone(timezone_name):
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%dT%H:%M:%S%z")


def _artifact_path_for_task(task):
    if task.get("artifact_path"):
        return str(task["artifact_path"])
    fixture_repo_name = Path(str(task["fixture_repo"])).name
    if fixture_repo_name not in TASK_FIXTURE_ARTIFACTS:
        raise ValueError(f"unsupported fixture repo for artifact lookup: {fixture_repo_name}")
    return TASK_FIXTURE_ARTIFACTS[fixture_repo_name]


def _run_verifier(command, cwd):
    """Run the fixed Python verifier form portably on Windows and POSIX."""
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = []
    if argv and Path(argv[0]).name.lower() in {"python", "python3", "py"} and "-c" in argv:
        if Path(argv[0]).name.lower() == "py" and len(argv) > 1 and argv[1] == "-3":
            argv.pop(1)
        argv[0] = sys.executable
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    return subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )


def _workspace_relative(path, workspace_root):
    return str(Path(path).resolve().relative_to(Path(workspace_root).resolve()))


def _scripted_outputs_for_task(task):
    outputs = SCRIPTED_MODEL_OUTPUTS.get(task["id"])
    if outputs is None:
        raise ValueError(f"no scripted model outputs for benchmark task: {task['id']}")
    return list(outputs)


def _fixture_snapshot_id(fixture_paths):
    sha = hashlib.sha256()
    ignored_parts = {".pico", "__pycache__"}
    for fixture_path in sorted({Path(path).resolve() for path in fixture_paths}, key=lambda path: str(path)):
        for path in sorted(
            (
                item
                for item in fixture_path.rglob("*")
                if item.is_file()
                and not ignored_parts.intersection(item.relative_to(fixture_path).parts)
            ),
            key=lambda item: str(item.relative_to(fixture_path)),
        ):
            sha.update(str(fixture_path.name).encode("utf-8"))
            sha.update(b"\0")
            sha.update(str(path.relative_to(fixture_path)).encode("utf-8"))
            sha.update(b"\0")
            sha.update(path.read_bytes())
            sha.update(b"\0")
    return "sha256:" + sha.hexdigest()


def validate_benchmark(data, repo_root=None):
    if not isinstance(data, dict):
        raise ValueError("benchmark must be a mapping")

    missing = [key for key in REQUIRED_BENCHMARK_KEYS if key not in data]
    if missing:
        raise ValueError(f"benchmark is missing required keys: {', '.join(missing)}")

    if int(data.get("schema_version", 0)) != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema_version")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("benchmark tasks must be a non-empty list")

    repo_root = Path(repo_root or Path.cwd()).resolve()
    seen_ids = set()
    normalized_tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"benchmark task at index {index} must be a mapping")

        missing_task_keys = [key for key in REQUIRED_TASK_KEYS if key not in task]
        if missing_task_keys:
            raise ValueError(
                f"benchmark task {task.get('id', index)!r} is missing required keys: {', '.join(missing_task_keys)}"
            )

        task_id = str(task["id"]).strip()
        if not task_id:
            raise ValueError(f"benchmark task at index {index} has an empty id")
        if task_id in seen_ids:
            raise ValueError(f"duplicate benchmark task id: {task_id}")
        seen_ids.add(task_id)

        fixture_repo = repo_root / str(task["fixture_repo"])
        if not fixture_repo.is_dir():
            raise ValueError(f"benchmark task {task_id} fixture repo does not exist: {task['fixture_repo']}")

        allowed_tools = task["allowed_tools"]
        if not isinstance(allowed_tools, list) or not allowed_tools:
            raise ValueError(f"benchmark task {task_id} allowed_tools must be a non-empty list")
        valid_tools = legal_tool_names()
        normalized_allowed_tools = []
        for tool in allowed_tools:
            tool_name = str(tool).strip()
            if not tool_name:
                raise ValueError(f"benchmark task {task_id} has an empty allowed_tools entry")
            if tool_name not in valid_tools:
                raise ValueError(f"benchmark task {task_id} has an unknown allowed_tools entry: {tool_name}")
            normalized_allowed_tools.append(tool_name)

        step_budget = int(task["step_budget"])
        if step_budget < 1:
            raise ValueError(f"benchmark task {task_id} step_budget must be positive")

        normalized_task = dict(task)
        normalized_task["id"] = task_id
        normalized_task["prompt"] = str(task["prompt"]).strip()
        normalized_task["fixture_repo"] = str(task["fixture_repo"]).strip()
        normalized_task["allowed_tools"] = normalized_allowed_tools
        normalized_task["step_budget"] = step_budget
        normalized_task["expected_artifact"] = str(task["expected_artifact"]).strip()
        normalized_task["verifier"] = str(task["verifier"]).strip()
        normalized_task["category"] = str(task["category"]).strip()
        normalized_tasks.append(normalized_task)

    normalized = dict(data)
    normalized["schema_version"] = BENCHMARK_SCHEMA_VERSION
    normalized["tasks"] = normalized_tasks
    return normalized


def load_benchmark(path=DEFAULT_BENCHMARK_PATH, repo_root=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if repo_root is None:
        repo_root = path.resolve().parent.parent
    return validate_benchmark(data, repo_root=repo_root)


def summarize_rows(rows):
    rows = list(rows)
    passed = sum(1 for row in rows if row.get("passed") or row.get("status") == "pass")
    failed = len(rows) - passed
    failure_category_counts = {}
    for row in rows:
        if row.get("passed") or row.get("status") == "pass":
            continue
        category = str(row.get("failure_category") or "unknown")
        failure_category_counts[category] = failure_category_counts.get(category, 0) + 1

    total_tasks = len(rows)
    within_budget = sum(1 for row in rows if row.get("within_budget"))
    verifier_passes = sum(1 for row in rows if row.get("verifier_passed"))
    return {
        "total_tasks": total_tasks,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total_tasks) if total_tasks else 0.0,
        "within_budget": within_budget,
        "verifier_passes": verifier_passes,
        "within_budget_rate": (within_budget / total_tasks) if total_tasks else 0.0,
        "verifier_pass_rate": (verifier_passes / total_tasks) if total_tasks else 0.0,
        "failure_category_counts": failure_category_counts,
    }


def _checkpoint_payload(
    checkpoint_id,
    current_goal,
    next_step,
    runtime_identity,
    *,
    schema_version=BENCHMARK_SCHEMA_VERSION,
    current_blocker="",
    key_files=None,
    freshness=None,
    summary="",
):
    return {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": "",
        "schema_version": "phase1-v1" if schema_version == BENCHMARK_SCHEMA_VERSION else str(schema_version),
        "created_at": "2026-04-15T08:00:00+00:00",
        "current_goal": current_goal,
        "completed": [],
        "excluded": [],
        "current_blocker": current_blocker,
        "next_step": next_step,
        "key_files": list(key_files or []),
        "freshness": dict(freshness or {}),
        "summary": summary or current_goal,
        "runtime_identity": dict(runtime_identity),
    }


def _apply_task_setup(agent, task, fixture_copy_root):
    setup = dict(task.get("setup", {}) or {})
    if not setup:
        return

    kind = str(setup.get("kind", "")).strip()
    if kind == "context_reduction":
        history_count = int(setup.get("history_count", 12))
        note_count = int(setup.get("note_count", 6))
        for index in range(history_count):
            agent.record(
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"benchmark-history-{index}-" + ("A" * 220),
                    "created_at": f"2026-04-15T09:{index:02d}:00+00:00",
                }
            )
        for index in range(note_count):
            agent.memory.append_note(
                f"benchmark-note-{index}-" + ("B" * 180),
                tags=("recall",),
                created_at=f"2026-04-15T10:{index:02d}:00+00:00",
            )
        agent.session["memory"] = agent.memory.to_dict()
        agent.context_manager.total_budget = int(setup.get("total_budget", 900))
        agent.context_manager.section_budgets = dict(
            setup.get(
                "section_budgets",
                {"prefix": 120, "memory": 120, "relevant_memory": 120, "history": 160},
            )
        )
        return

    if kind == "freshness_mismatch":
        path = str(setup.get("path", "sample.txt"))
        summary_text = str(setup.get("summary", f"{path}: stale benchmark summary"))
        agent.memory.set_file_summary(path, summary_text)
        agent.memory.remember_file(path)
        freshness = agent.memory.to_dict()["file_summaries"][path]["freshness"]
        agent.session["memory"] = agent.memory.to_dict()
        agent.session["checkpoints"] = {
            "current_id": "ckpt_freshness",
            "items": {
                "ckpt_freshness": _checkpoint_payload(
                    "ckpt_freshness",
                    current_goal="Re-anchor stale benchmark file state",
                    next_step=f"Re-read {path}",
                    runtime_identity={"workspace_fingerprint": agent.workspace.fingerprint()},
                    key_files=[{"path": path, "freshness": freshness}],
                    freshness={path: freshness},
                    summary="stale benchmark checkpoint",
                )
            },
        }
        agent.session_store.save(agent.session)
        (fixture_copy_root / path).write_text(str(setup.get("mutated_text", "alpha\nbeta\nstale-updated\nplaceholder\n")), encoding="utf-8")
        return

    if kind == "workspace_mismatch":
        agent.session["checkpoints"] = {
            "current_id": "ckpt_workspace",
            "items": {
                "ckpt_workspace": _checkpoint_payload(
                    "ckpt_workspace",
                    current_goal="Recover after benchmark workspace drift",
                    next_step="Rebuild runtime state from a fresh checkpoint",
                    runtime_identity={"workspace_fingerprint": "outdated-benchmark-fingerprint"},
                    summary="workspace drift benchmark checkpoint",
                )
            },
        }
        agent.session_store.save(agent.session)
        return


class BenchmarkEvaluator:
    def __init__(
        self,
        benchmark_path=DEFAULT_BENCHMARK_PATH,
        artifact_path=DEFAULT_ARTIFACT_PATH,
        workspace_root=None,
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        timezone_name=DEFAULT_TIMEZONE,
        model_client_factory=None,
        action_chunking=None,
    ):
        self.benchmark_path = Path(benchmark_path)
        self.artifact_path = Path(artifact_path)
        self.workspace_root = Path(workspace_root) if workspace_root is not None else Path(
            tempfile.mkdtemp(prefix="pico-benchmark-")
        )
        self.model_name = model_name
        self.model_version = model_version
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timezone_name = timezone_name
        self.model_client_factory = model_client_factory
        self.action_chunking = normalize_action_chunking(action_chunking)
        self.repo_root = self.benchmark_path.resolve().parent.parent

    def load(self):
        return load_benchmark(self.benchmark_path, repo_root=self.repo_root)

    def run(self, task_ids=None, continue_on_error=False):
        benchmark = self.load()
        tasks = benchmark["tasks"]
        if task_ids is not None:
            requested = [str(task_id) for task_id in task_ids]
            requested_set = set(requested)
            known = {task["id"] for task in tasks}
            unknown = sorted(requested_set - known)
            if unknown:
                raise ValueError("unknown benchmark task ids: " + ", ".join(unknown))
            tasks = [task for task in tasks if task["id"] in requested_set]
        rows = []
        for task in tasks:
            try:
                rows.append(self.run_task(task))
            except Exception as exc:
                if not continue_on_error:
                    raise
                rows.append(
                    {
                        "id": task["id"],
                        "category": task["category"],
                        "status": "fail",
                        "passed": False,
                        "failure_category": "runner_exception",
                        "within_budget": False,
                        "verifier_passed": False,
                        "expected_artifact_exists": False,
                        "non_failure_stop_reason": False,
                        "stop_reason": "runner_exception",
                        "error_type": type(exc).__name__,
                        "error": "task execution failed; inspect the per-task run evidence",
                        "logical_decisions": None,
                        "provider_requests": None,
                        "provider_retries": None,
                        "provider_responses": None,
                        "provider_attempts": [],
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "cached_tokens": None,
                        "token_usage_coverage": 0.0,
                        "chunk_count": 0,
                        "chunk_lengths": [],
                        "chunk_mean_length": 0.0,
                        "chunk_max_length": 0,
                        "chunk_interrupts": 0,
                        "chunk_interrupt_rate": 0.0,
                        "chunk_stop_distribution": {},
                        "e2e_latency_ms": None,
                    }
                )
        summary = summarize_rows(rows)
        artifact = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "captured_at": _now_in_timezone(self.timezone_name),
            "runtime": {
                "commit_sha": _git_value(["rev-parse", "HEAD"], cwd=self.repo_root),
                "branch": _git_value(["branch", "--show-current"], cwd=self.repo_root),
            },
            "benchmark": {
                "source": str(self.benchmark_path.resolve().relative_to(self.repo_root)),
                "task_count": len(tasks),
            },
            "reproducibility": {
                "fixture_snapshot_id": _fixture_snapshot_id(
                    self.repo_root / str(task["fixture_repo"]) for task in tasks
                ),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "decoding": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                },
                "timezone": self.timezone_name,
                "locale": _current_locale(),
                "task_ids": [task["id"] for task in tasks],
                "action_chunking": dict(self.action_chunking),
            },
            "summary": summary,
            "failure_category_counts": summary["failure_category_counts"],
            "rows": rows,
        }
        self._write_artifact(artifact)
        return artifact

    def run_task(self, task):
        task = dict(task)
        fixture_source = self.repo_root / task["fixture_repo"]
        fixture_copy_root = self.workspace_root / task["id"] / fixture_source.name
        if fixture_copy_root.exists():
            shutil.rmtree(fixture_copy_root)
        fixture_copy_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fixture_source, fixture_copy_root)

        workspace = WorkspaceContext.build(
            fixture_copy_root,
            repo_root_override=fixture_copy_root,
        )
        session_store = SessionStore(fixture_copy_root / ".pico" / "sessions")
        run_store = RunStore(fixture_copy_root / ".pico" / "runs")
        if self.model_client_factory is not None:
            model_client = self.model_client_factory(task=task, workspace=workspace)
        else:
            model_client = FakeModelClient(_scripted_outputs_for_task(task))
        agent = Pico(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            run_store=run_store,
            approval_policy="auto",
            max_steps=int(task["step_budget"]),
            max_new_tokens=self.max_new_tokens,
            allowed_tools=task["allowed_tools"],
            action_chunking=self.action_chunking,
        )
        _apply_task_setup(agent, task, fixture_copy_root)

        initial_history_empty = len(agent.session["history"]) == 0
        initial_memory_state = agent.memory.to_dict()
        initial_memory_empty = memorylib.is_effectively_empty(initial_memory_state)
        initial_task_summary_empty = not str(initial_memory_state["working"]["task_summary"]).strip()
        initial_episodic_notes_empty = not initial_memory_state["episodic_notes"]

        run_started_at = time.monotonic()
        final_answer = agent.ask(task["prompt"])
        e2e_latency_ms = int((time.monotonic() - run_started_at) * 1000)
        task_state = agent.current_task_state
        run_dir = Path(agent.current_run_dir)
        task_state_path = agent.run_store.task_state_path(task_state)
        report_path = agent.run_store.report_path(task_state)
        report = agent.run_store.load_report(task_state.run_id)
        trace_events = [
            json.loads(line)
            for line in agent.run_store.trace_path(task_state).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunk_summaries = [
            event for event in trace_events if event.get("event") == "chunk_finished"
        ]
        chunk_stop_distribution = {}
        for chunk_summary in chunk_summaries:
            reason = str(chunk_summary.get("stop_reason") or "completed")
            chunk_stop_distribution[reason] = chunk_stop_distribution.get(reason, 0) + 1

        artifact_path = _artifact_path_for_task(task)
        artifact_file = fixture_copy_root / artifact_path
        expected_artifact_exists = artifact_file.exists()
        artifact_digest = _digest_file(artifact_file) if expected_artifact_exists else ""

        verifier = _run_verifier(task["verifier"], fixture_copy_root)

        within_budget = task_state.tool_steps <= int(task["step_budget"])
        verifier_passed = verifier.returncode == 0
        non_failure_stop_reason = task_state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
        passed = within_budget and verifier_passed and expected_artifact_exists and non_failure_stop_reason
        failure_category = None if passed else self._failure_category(
            within_budget=within_budget,
            verifier_passed=verifier_passed,
            expected_artifact_exists=expected_artifact_exists,
            non_failure_stop_reason=non_failure_stop_reason,
        )

        provider_attempts = [dict(attempt) for attempt in task_state.provider_attempts]
        token_fields = ("input_tokens", "output_tokens", "total_tokens")
        token_usage_complete = bool(provider_attempts) and all(
            all(attempt.get(field) is not None for field in token_fields)
            for attempt in provider_attempts
        )
        token_usage_coverage = (
            sum(
                all(attempt.get(field) is not None for field in token_fields)
                for attempt in provider_attempts
            )
            / len(provider_attempts)
            if provider_attempts
            else 0.0
        )
        token_totals = {
            field: sum(int(attempt[field]) for attempt in provider_attempts)
            if token_usage_complete and all(attempt.get(field) is not None for attempt in provider_attempts)
            else None
            for field in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")
        }

        return {
            "id": task["id"],
            "prompt": task["prompt"],
            "fixture_repo": task["fixture_repo"],
            "fixture_copy_relpath": _workspace_relative(fixture_copy_root, self.workspace_root),
            "run_id": task_state.run_id,
            "run_dir_relpath": _workspace_relative(run_dir, self.workspace_root),
            "task_state_relpath": _workspace_relative(task_state_path, self.workspace_root),
            "report_relpath": _workspace_relative(report_path, self.workspace_root),
            "allowed_tools": list(task["allowed_tools"]),
            "step_budget": int(task["step_budget"]),
            "expected_artifact": task["expected_artifact"],
            "artifact_path": artifact_path,
            "artifact_exists": expected_artifact_exists,
            "artifact_digest": artifact_digest,
            "verifier": task["verifier"],
            "verifier_exit_code": verifier.returncode,
            "verifier_stdout": verifier.stdout,
            "verifier_stderr": verifier.stderr,
            "category": task["category"],
            "status": "pass" if passed else "fail",
            "passed": passed,
            "failure_category": failure_category,
            "within_budget": within_budget,
            "verifier_passed": verifier_passed,
            "expected_artifact_exists": expected_artifact_exists,
            "non_failure_stop_reason": non_failure_stop_reason,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "logical_decisions": task_state.logical_decisions,
            "provider_requests": task_state.provider_requests,
            "provider_retries": task_state.provider_retries,
            "provider_responses": task_state.provider_responses,
            "provider_attempts": provider_attempts,
            "provider_attempt_count": len(provider_attempts),
            "input_tokens": token_totals["input_tokens"],
            "output_tokens": token_totals["output_tokens"],
            "total_tokens": token_totals["total_tokens"],
            "cached_tokens": token_totals["cached_tokens"],
            "token_usage_coverage": token_usage_coverage,
            "usage_missing_responses": task_state.usage_missing_responses,
            "usage_missing_rate": (
                task_state.usage_missing_responses / task_state.provider_responses
                if task_state.provider_responses
                else 0.0
            ),
            "auxiliary_requests": task_state.auxiliary_requests,
            "primitive_tool_calls": task_state.primitive_tool_calls,
            "primitive_submissions": task_state.primitive_submissions,
            "executed_tool_calls": task_state.executed_tool_calls,
            "successful_tool_calls": task_state.successful_tool_calls,
            "failed_tool_calls": task_state.failed_tool_calls,
            "rejected_tool_calls": task_state.rejected_tool_calls,
            "unknown_tool_calls": task_state.unknown_tool_calls,
            "chunk_count": task_state.chunk_count,
            "chunk_interrupt_rate": (
                task_state.chunk_interrupts / task_state.chunk_count
                if task_state.chunk_count
                else 0.0
            ),
            "chunk_lengths": list(task_state.chunk_lengths),
            "chunk_mean_length": (
                sum(task_state.chunk_lengths) / len(task_state.chunk_lengths)
                if task_state.chunk_lengths
                else 0.0
            ),
            "chunk_max_length": max(task_state.chunk_lengths, default=0),
            "chunk_interrupts": task_state.chunk_interrupts,
            "chunk_stop_distribution": chunk_stop_distribution,
            "final_answer": final_answer,
            "stop_reason": task_state.stop_reason,
            "e2e_latency_ms": e2e_latency_ms,
            "initial_history_empty": initial_history_empty,
            "initial_memory_empty": initial_memory_empty,
            "initial_task_summary_empty": initial_task_summary_empty,
            "initial_episodic_notes_empty": initial_episodic_notes_empty,
            "task_state": task_state.to_dict(),
            "report": report,
        }

    def _failure_category(
        self,
        within_budget,
        verifier_passed,
        expected_artifact_exists,
        non_failure_stop_reason,
    ):
        if not expected_artifact_exists:
            return "missing_artifact"
        if not within_budget:
            return "budget_exceeded"
        if not verifier_passed:
            return "verifier_failed"
        if not non_failure_stop_reason:
            return "failure_stop_reason"
        return "unknown"

    def _write_artifact(self, artifact):
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_file(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fixed_benchmark(
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=DEFAULT_ARTIFACT_PATH,
    workspace_root=None,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    timezone_name=DEFAULT_TIMEZONE,
    model_client_factory=None,
    action_chunking=None,
):
    evaluator = BenchmarkEvaluator(
        benchmark_path=benchmark_path,
        artifact_path=artifact_path,
        workspace_root=workspace_root,
        model_name=model_name,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timezone_name=timezone_name,
        model_client_factory=model_client_factory,
        action_chunking=action_chunking,
    )
    return evaluator.run()


def run_harness_regression_v2(
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=DEFAULT_HARNESS_REGRESSION_V2_ARTIFACT_PATH,
    workspace_root=None,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    timezone_name=DEFAULT_TIMEZONE,
    model_client_factory=None,
):
    return run_fixed_benchmark(
        benchmark_path=benchmark_path,
        artifact_path=artifact_path,
        workspace_root=workspace_root,
        model_name=model_name,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timezone_name=timezone_name,
        model_client_factory=model_client_factory,
    )
