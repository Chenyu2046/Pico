"""Checkpoint and resume-state helpers."""

import uuid

from .features import memory as memorylib
from .workspace import clip, now

CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
CHECKPOINT_NONE_STATUS = "no-checkpoint"
CHECKPOINT_FULL_VALID_STATUS = "full-valid"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"

RUNTIME_IDENTITY_KEYS = (
    "cwd",
    "model",
    "model_client",
    "approval_policy",
    "read_only",
    "max_steps",
    "max_new_tokens",
    "feature_flags",
    "shell_env_allowlist",
    "workspace_fingerprint",
    "tool_signature",
    "action_chunking",
)


def current_runtime_identity(agent):
    return {
        "session_id": agent.session.get("id", ""),
        "cwd": str(agent.root),
        "model": str(getattr(agent.model_client, "model", "")),
        "model_client": agent.model_client.__class__.__name__,
        "approval_policy": agent.approval_policy,
        "read_only": bool(agent.read_only),
        "max_steps": int(agent.max_steps),
        "max_new_tokens": int(agent.max_new_tokens),
        "feature_flags": dict(agent.feature_flags),
        "shell_env_allowlist": list(agent.shell_env_allowlist),
        "workspace_fingerprint": getattr(getattr(agent, "prefix_state", None), "workspace_fingerprint", agent.workspace.fingerprint()),
        "tool_signature": agent.tool_signature(),
        "action_chunking": dict(getattr(agent, "action_chunking", {}) or {}),
    }


def checkpoint_state(agent):
    agent._ensure_session_shape()
    return agent.session["checkpoints"]


def current_checkpoint(agent):
    state = checkpoint_state(agent)
    checkpoint_id = str(state.get("current_id", "")).strip()
    if not checkpoint_id:
        return None
    return state.get("items", {}).get(checkpoint_id)


def latest_committed_action(agent):
    """Return the highest committed primitive action checkpoint, if any."""
    items = checkpoint_state(agent).get("items", {})
    committed = [
        item
        for item in items.values()
        if _is_complete_committed_action(item)
    ]
    return max(committed, key=lambda item: int(item.get("action_seq", 0) or 0), default=None)


def _is_complete_committed_action(item):
    if not isinstance(item, dict):
        return False
    if item.get("committed") is not True or item.get("commit_status") != "committed":
        return False
    if item.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False
    checkpoint_id = str(item.get("checkpoint_id", "")).strip()
    action_id = str(item.get("action_id", "")).strip()
    try:
        action_seq = int(item.get("action_seq", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        checkpoint_id
        and action_id
        and action_seq > 0
    )


def reconcile_session_to_watermark(agent):
    """Reconcile action history, checkpoints, and current_id to the watermark."""
    committed = latest_committed_action(agent)
    watermark = int(committed.get("action_seq", 0) or 0) if committed else 0

    discarded_actions = set()
    history = list(agent.session.get("history", []))
    kept = []
    for item in history:
        try:
            action_seq = int(item.get("action_seq", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            action_seq = 0
        if item.get("action_id") and action_seq > watermark:
            discarded_actions.add((str(item.get("action_id")), action_seq))
            continue
        kept.append(item)
    if len(kept) != len(history):
        agent.session["history"] = kept

    checkpoints = checkpoint_state(agent)
    items = dict(checkpoints.get("items", {}))
    current_id = str(checkpoints.get("current_id", "")).strip()
    current = items.get(current_id)
    for checkpoint_id, item in list(items.items()):
        if not isinstance(item, dict) or not item.get("action_id"):
            continue
        try:
            action_seq = int(item.get("action_seq", 0) or 0)
        except (TypeError, ValueError):
            action_seq = 0
        if action_seq > watermark and not _is_complete_committed_action(item):
            discarded_actions.add((str(item.get("action_id")), action_seq))
            items.pop(checkpoint_id, None)
    checkpoints["items"] = items

    def legal_checkpoint(checkpoint):
        if not isinstance(checkpoint, dict):
            return False
        if not checkpoint.get("action_id"):
            return True
        try:
            action_seq = int(checkpoint.get("action_seq", 0) or 0)
        except (TypeError, ValueError):
            return False
        return action_seq <= watermark and _is_complete_committed_action(checkpoint)

    if not legal_checkpoint(items.get(current_id)):
        fallback_id = str(current.get("parent_checkpoint_id", "")).strip() if current else ""
        seen = set()
        while fallback_id and fallback_id not in seen:
            seen.add(fallback_id)
            candidate = items.get(fallback_id)
            if legal_checkpoint(candidate):
                break
            fallback_id = str(candidate.get("parent_checkpoint_id", "")).strip() if candidate else ""
        else:
            fallback_id = ""
        if not fallback_id:
            legal_ids = [
                checkpoint_id
                for checkpoint_id, item in items.items()
                if legal_checkpoint(item)
            ]
            fallback_id = next(
                (
                    checkpoint_id
                    for checkpoint_id in reversed(legal_ids)
                    if _is_complete_committed_action(items[checkpoint_id])
                ),
                legal_ids[-1] if legal_ids else "",
            )
        checkpoints["current_id"] = fallback_id

    return len(discarded_actions)


def evaluate_resume_state(agent):
    previous_resume_state = dict(agent.session.get("resume_state", {}) or {})
    recovery_tail_discarded = reconcile_session_to_watermark(agent)
    recovery_tail_discarded_cumulative = int(
        previous_resume_state.get(
            "recovery_tail_discarded_cumulative",
            previous_resume_state.get("recovery_tail_discarded", 0),
        )
        or 0
    ) + recovery_tail_discarded
    invalidated = agent.invalidate_stale_memory()
    checkpoint = current_checkpoint(agent)
    committed_action = latest_committed_action(agent)
    status = CHECKPOINT_NONE_STATUS
    stale_paths = list(invalidated)
    mismatch_fields = []
    if checkpoint:
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
        else:
            for item in checkpoint.get("key_files", []):
                path = str(item.get("path", "")).strip()
                if not path:
                    continue
                expected = item.get("freshness")
                current = memorylib.file_freshness(path, agent.root)
                if expected != current and path not in stale_paths:
                    stale_paths.append(path)
            saved_identity = dict(checkpoint.get("runtime_identity", {}) or agent.session.get("runtime_identity", {}) or {})
            current_identity = current_runtime_identity(agent)
            for key in RUNTIME_IDENTITY_KEYS:
                if key not in saved_identity:
                    continue
                if saved_identity.get(key) != current_identity.get(key):
                    mismatch_fields.append(key)
            mismatch_fields.sort()
            if stale_paths:
                status = CHECKPOINT_PARTIAL_STALE_STATUS
            elif mismatch_fields:
                status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
            else:
                status = CHECKPOINT_FULL_VALID_STATUS

    resume_state = {
        "status": status,
        "stale_paths": stale_paths,
        "runtime_identity_mismatch_fields": mismatch_fields,
        "recovery_tail_discarded": recovery_tail_discarded,
        "recovery_tail_discarded_cumulative": recovery_tail_discarded_cumulative,
        "last_committed_action": {
            "action_id": str(committed_action.get("action_id", "")),
            "action_seq": int(committed_action.get("action_seq", 0) or 0),
            "action_status": str(committed_action.get("action_status", "")),
            "checkpoint_id": str(committed_action.get("checkpoint_id", "")),
        }
        if committed_action
        else {},
        "stale_summary_invalidations": max(
            len(invalidated),
            int(previous_resume_state.get("stale_summary_invalidations", 0))
            if status == CHECKPOINT_PARTIAL_STALE_STATUS
            else 0,
        ),
    }
    agent.session["resume_state"] = resume_state
    agent.session["runtime_identity"] = current_runtime_identity(agent)
    return resume_state


def render_checkpoint_text(agent):
    checkpoint = current_checkpoint(agent)
    if not checkpoint:
        return ""
    lines = [
        "Task checkpoint:",
        f"- Resume status: {agent.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
        f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
        f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
        f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
    ]
    key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
    lines.append(f"- Key files: {', '.join(key_files) or '-'}")
    if checkpoint.get("completed"):
        lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
    if checkpoint.get("excluded"):
        lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))
    if agent.resume_state.get("stale_paths"):
        lines.append("- Stale paths: " + ", ".join(agent.resume_state["stale_paths"]))
    summary = str(checkpoint.get("summary", "")).strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    committed_action = agent.resume_state.get("last_committed_action", {})
    if committed_action:
        lines.append(
            "- Last committed action: "
            f"{committed_action.get('action_seq')} {committed_action.get('action_status')}"
        )
    return "\n".join(lines)


def infer_next_step(task_state):
    if task_state.status == "completed":
        return "No next step recorded."
    if task_state.stop_reason == "step_limit_reached":
        return "Resume from the latest checkpoint and continue the task."
    if task_state.last_tool:
        return f"Decide the next action after {task_state.last_tool}."
    return "Continue the task from the latest checkpoint."


def build_checkpoint(agent, task_state, user_message, trigger, action_result=None):
    current = current_checkpoint(agent)
    checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
    key_files = []
    freshness = {}
    for path in agent.memory.to_dict()["working"]["recent_files"]:
        file_freshness = memorylib.file_freshness(path, agent.root)
        freshness[path] = file_freshness
        key_files.append({"path": path, "freshness": file_freshness})
    action_committed = bool(
        action_result is not None
        and action_result.result_known
        and action_result.commit_eligible
    )
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": now(),
        "current_goal": str(user_message),
        "completed": [task_state.final_answer] if task_state.final_answer else [],
        "excluded": [],
        "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
        "next_step": infer_next_step(task_state),
        "key_files": key_files,
        "freshness": freshness,
        "summary": f"{trigger}: {clip(str(user_message), 120)}",
        "runtime_identity": current_runtime_identity(agent),
        "committed": action_committed,
        "commit_status": (
            "committed"
            if action_committed
            else ("not-committed" if action_result is not None else "not-applicable")
        ),
        "action_id": str(getattr(action_result, "action_id", "") or ""),
        "action_seq": int(getattr(action_result, "action_seq", task_state.action_seq) or 0),
        "action_status": str(getattr(action_result, "status", "") or ""),
        "result_known": bool(getattr(action_result, "result_known", True)) if action_result is not None else True,
        "commit_eligible": bool(getattr(action_result, "commit_eligible", True)) if action_result is not None else True,
    }
    return checkpoint


def attach_checkpoint(agent, task_state, checkpoint):
    state = checkpoint_state(agent)
    state["items"][checkpoint["checkpoint_id"]] = checkpoint
    state["current_id"] = checkpoint["checkpoint_id"]
    task_state.checkpoint_id = checkpoint["checkpoint_id"]
    agent.session["runtime_identity"] = checkpoint["runtime_identity"]
    return checkpoint


def create_checkpoint(agent, task_state, user_message, trigger, action_result=None):
    checkpoint = build_checkpoint(
        agent,
        task_state,
        user_message,
        trigger,
        action_result=action_result,
    )
    attach_checkpoint(agent, task_state, checkpoint)
    agent.session_path = agent.session_store.save(agent.session)
    return checkpoint
