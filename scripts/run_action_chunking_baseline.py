"""Run the ReAct A baseline from an original Pico worktree.

This helper intentionally imports Pico from its own worktree instead of the
caller repository, so a disabled chunk flag on the final commit cannot be
mistaken for the original baseline.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pico import Pico, SessionStore, WorkspaceContext
from pico.evaluation.token_usage import aggregate_provider_usage
from pico.providers.clients import OpenAICompatibleModelClient
from pico.run_store import RunStore


def _git_value(args, fallback=""):
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or fallback


def _redact(value, secret):
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _run_verifier(command, cwd):
    argv = shlex.split(command)
    if argv and Path(argv[0]).name.lower() in {"python", "python3", "py"} and "-c" in argv:
        if Path(argv[0]).name.lower() == "py" and len(argv) > 1 and argv[1] == "-3":
            argv.pop(1)
        argv[0] = sys.executable
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    return subprocess.run(command, cwd=cwd, shell=True, capture_output=True, text=True, check=False)


def _digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_error(value):
    text = str(value)
    secret = os.environ.get("PICO_OPENAI_API_KEY", "")
    return text.replace(secret, "<redacted>")[:1000] if secret else text[:1000]


def _summary(rows):
    passed = sum(bool(row.get("passed")) for row in rows)
    verifier_passes = sum(bool(row.get("verifier_passed")) for row in rows)
    within_budget = sum(bool(row.get("within_budget")) for row in rows)
    total = len(rows)
    return {
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "verifier_passes": verifier_passes,
        "verifier_pass_rate": verifier_passes / total if total else 0.0,
        "within_budget": within_budget,
        "within_budget_rate": within_budget / total if total else 0.0,
    }


def run(benchmark_path, output_path, workspace_root, api_key, base_url, model, temperature, max_new_tokens, timeout, task_ids=None):
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    if task_ids is not None:
        task_ids = set(task_ids)
        benchmark["tasks"] = [task for task in benchmark["tasks"] if task["id"] in task_ids]
    rows = []
    for task in benchmark["tasks"]:
        fixture_source = REPO_ROOT / task["fixture_repo"]
        fixture_copy = Path(workspace_root) / task["id"] / fixture_source.name
        if fixture_copy.exists():
            import shutil

            shutil.rmtree(fixture_copy)
        fixture_copy.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copytree(fixture_source, fixture_copy)
        workspace = WorkspaceContext.build(fixture_copy, repo_root_override=fixture_copy)
        session_store = SessionStore(fixture_copy / ".pico" / "sessions")
        run_store = RunStore(fixture_copy / ".pico" / "runs")
        client = OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
        )
        agent = Pico(
            model_client=client,
            workspace=workspace,
            session_store=session_store,
            run_store=run_store,
            approval_policy="auto",
            max_steps=int(task["step_budget"]),
            max_new_tokens=max_new_tokens,
            allowed_tools=task["allowed_tools"],
            action_chunking={"enabled": False},
            secret_env_names=["PICO_OPENAI_API_KEY"],
        )
        started_at = time.monotonic()
        try:
            final_answer = agent.ask(task["prompt"])
        except Exception as exc:  # noqa: BLE001 - record one failed task and continue the fixed set
            rows.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "status": "error",
                    "passed": False,
                    "verifier_passed": False,
                    "within_budget": False,
                    "expected_artifact_exists": False,
                    "non_failure_stop_reason": False,
                    "stop_reason": "runner_exception",
                    "error_type": type(exc).__name__,
                    "logical_decisions": None,
                    "provider_requests": None,
                    "provider_retries": None,
                    "provider_responses": None,
                    "provider_attempts": [],
                    "provider_attempt_count": 0,
                    "primitive_submissions": 0,
                    "chunk_count": 0,
                    "chunk_interrupts": 0,
                    "chunk_interrupt_rate": 0.0,
                    "chunk_lengths": [],
                    "chunk_mean_length": 0.0,
                    "chunk_max_length": 0,
                    "chunk_stop_distribution": {},
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "cached_tokens": None,
                    "token_usage_coverage": 0.0,
                    "e2e_latency_ms": int((time.monotonic() - started_at) * 1000),
                    "error": _redact(str(exc)[:1000], api_key),
                }
            )
            continue
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        state = agent.current_task_state
        report = run_store.load_report(state.run_id)
        artifact_path = fixture_copy / str(task.get("artifact_path", "README.md"))
        verifier = _run_verifier(task["verifier"], fixture_copy)
        within_budget = state.tool_steps <= int(task["step_budget"])
        verifier_passed = verifier.returncode == 0
        completed = state.stop_reason == "final_answer_returned"
        passed = within_budget and verifier_passed and artifact_path.exists() and completed
        attempts = [dict(item) for item in getattr(state, "provider_attempts", [])]
        token_usage = aggregate_provider_usage(attempts)
        rows.append(
            {
                "id": task["id"],
                "category": task["category"],
                "status": "pass" if passed else "fail",
                "passed": passed,
                "verifier_passed": verifier_passed,
                "within_budget": within_budget,
                "stop_reason": state.stop_reason,
                "tool_steps": state.tool_steps,
                "attempts": state.attempts,
                "logical_decisions": state.logical_decisions,
                "provider_requests": state.provider_requests,
                "provider_retries": state.provider_retries,
                "provider_responses": state.provider_responses,
                "provider_attempts": attempts,
                "provider_attempt_count": len(attempts),
                "primitive_submissions": getattr(state, "primitive_submissions", state.tool_steps),
                "chunk_count": state.chunk_count,
                "chunk_interrupts": state.chunk_interrupts,
                "chunk_interrupt_rate": state.chunk_interrupts / state.chunk_count if state.chunk_count else 0.0,
                "chunk_lengths": list(state.chunk_lengths),
                "chunk_mean_length": sum(state.chunk_lengths) / len(state.chunk_lengths) if state.chunk_lengths else 0.0,
                "chunk_max_length": max(state.chunk_lengths, default=0),
                "chunk_stop_distribution": {},
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "total_tokens": token_usage["total_tokens"],
                "cached_tokens": token_usage["cached_tokens"],
                "token_usage_coverage": token_usage["token_usage_coverage"],
                "token_usage_complete": token_usage["token_usage_complete"],
                "e2e_latency_ms": elapsed_ms,
                "final_answer": final_answer,
                "artifact_exists": artifact_path.exists(),
                "artifact_digest": _digest(artifact_path) if artifact_path.exists() else "",
                "verifier_exit_code": verifier.returncode,
                "verifier_stdout": verifier.stdout,
                "verifier_stderr": verifier.stderr,
                "report": report,
            }
        )
    artifact = {
        "schema_version": 1,
        "runtime": {"commit_sha": _git_value(["rev-parse", "HEAD"]), "branch": _git_value(["branch", "--show-current"])},
        "reproducibility": {"model_name": model, "decoding": {"temperature": temperature, "max_new_tokens": max_new_tokens}, "task_ids": [task["id"] for task in benchmark["tasks"]], "action_chunking": {"enabled": False}},
        "summary": _summary(rows),
        "rows": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = _redact(artifact, api_key)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--tasks", default="")
    args = parser.parse_args(argv)
    values = vars(args)
    values["benchmark_path"] = values.pop("benchmark")
    values["output_path"] = values.pop("output")
    values["workspace_root"] = values.pop("workspace")
    values["api_key"] = os.environ.get("PICO_OPENAI_API_KEY", "")
    task_ids = [item for item in values.pop("tasks").split(",") if item]
    values["task_ids"] = task_ids or None
    run(**values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
