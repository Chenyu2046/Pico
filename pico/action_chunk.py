"""Read-only action chunk planning, validation, and execution primitives."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from . import checkpoint as checkpointlib
from .action_commit import ActionCommitter
from .workspace import clip


READ_ONLY_CHUNK_TOOLS = ("list_files", "read_file", "search")
ACTION_STATUSES = ("planned", "accepted", "rejected", "started", "completed", "failed", "interrupted")
ACTION_TRANSITIONS = {
    "planned": {"accepted", "rejected"},
    "accepted": {"started", "rejected"},
    "started": {"completed", "failed", "rejected", "interrupted"},
    "rejected": set(),
    "completed": set(),
    "failed": set(),
    "interrupted": set(),
}
ACTION_CHUNK_KEYS = {
    "enabled",
    "max_actions_per_chunk",
    "allowed_tools",
    "observation_budget_chars",
    "skill_guidance_enabled",
}
DEFAULT_ACTION_CHUNKING = {
    "enabled": False,
    "max_actions_per_chunk": 4,
    "allowed_tools": list(READ_ONLY_CHUNK_TOOLS),
    "observation_budget_chars": 12000,
    "skill_guidance_enabled": False,
}
RESULT_REFERENCE_PATTERN = re.compile(r"\$\{\s*action_\d+\.result[^}]*\}")
STOP_REASON_PRIORITY = (
    "process_crash_or_unknown_result",
    "tool_failed",
    "permission_or_policy_violation",
    "budget_exhausted",
    "observation_budget_exceeded",
    "hard_boundary",
    "skill_boundary",
    "chunk_completed",
)


def normalize_action_chunking(config=None):
    """Validate the single supported ``action_chunking`` configuration root."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("action_chunking must be a mapping")
    unknown = sorted(str(key) for key in config if key not in ACTION_CHUNK_KEYS)
    if unknown:
        raise ValueError(f"unknown action_chunking fields: {', '.join(unknown)}")

    normalized = dict(DEFAULT_ACTION_CHUNKING)
    normalized.update(config)
    if not isinstance(normalized["enabled"], bool):
        raise ValueError("action_chunking.enabled must be a boolean")
    max_actions = normalized["max_actions_per_chunk"]
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
        raise ValueError("action_chunking.max_actions_per_chunk must be a positive integer")
    observation_budget = normalized["observation_budget_chars"]
    if isinstance(observation_budget, bool) or not isinstance(observation_budget, int) or observation_budget < 1:
        raise ValueError("action_chunking.observation_budget_chars must be a positive integer")
    if not isinstance(normalized["skill_guidance_enabled"], bool):
        raise ValueError("action_chunking.skill_guidance_enabled must be a boolean")

    allowed_tools = normalized["allowed_tools"]
    if not isinstance(allowed_tools, (list, tuple)) or not allowed_tools:
        raise ValueError("action_chunking.allowed_tools must be a non-empty list")
    allowed_tools = [str(name).strip() for name in allowed_tools]
    if any(not name for name in allowed_tools):
        raise ValueError("action_chunking.allowed_tools must contain non-empty names")
    invalid_tools = sorted(set(allowed_tools) - set(READ_ONLY_CHUNK_TOOLS))
    if invalid_tools:
        raise ValueError(
            "action_chunking.allowed_tools only supports read-only tools in phase 1: "
            + ", ".join(invalid_tools)
        )
    normalized["max_actions_per_chunk"] = max_actions
    normalized["observation_budget_chars"] = observation_budget
    normalized["allowed_tools"] = list(dict.fromkeys(allowed_tools))
    return normalized


@dataclass(frozen=True)
class Action:
    name: str
    args: dict = field(default_factory=dict)
    boundary_hint: bool = False

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("chunk action must be an object")
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("chunk action is missing a tool name")
        args = payload.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError(f"chunk action args for {name} must be an object")
        boundary_hint = payload.get("boundary_hint", False)
        if not isinstance(boundary_hint, bool):
            raise ValueError(f"chunk action boundary_hint for {name} must be a boolean")
        return cls(name=name, args=dict(args), boundary_hint=boundary_hint)


@dataclass
class ActionState:
    action: Action
    status: str = "planned"

    def transition(self, status):
        status = str(status)
        if status not in ACTION_STATUSES:
            raise ValueError(f"unknown action status: {status}")
        if status not in ACTION_TRANSITIONS[self.status]:
            raise ValueError(f"invalid action transition: {self.status} -> {status}")
        self.status = status
        return self


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[Action, ...]
    skill_id: str | None = None
    boundary_hint: bool = False
    chunk_id: str = ""

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("chunk payload must be an object")
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError("chunk payload actions must be a list")
        skill_id = payload.get("skill_id", payload.get("skill"))
        if skill_id is not None:
            skill_id = str(skill_id).strip() or None
        boundary_hint = payload.get("boundary_hint", False)
        if not isinstance(boundary_hint, bool):
            raise ValueError("chunk boundary_hint must be a boolean")
        return cls(
            actions=tuple(Action.from_payload(action) for action in actions),
            chunk_id=str(payload.get("chunk_id", "") or "").strip(),
            skill_id=skill_id,
            boundary_hint=boundary_hint,
        )


@dataclass(frozen=True)
class ChunkValidation:
    chunk: ActionChunk | None
    planned_actions: int
    accepted_actions: tuple[Action, ...] = ()
    rejected_actions: int = 0
    remaining_actions: int = 0
    terminal_status: str = "rejected"
    stop_reason: str = "permission_or_policy_violation"
    error: str = ""
    explicit_dependency_check: str = "passed"


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    action_seq: int
    action: Action
    status: str
    content: str
    metadata: dict
    checkpoint_id: str = ""
    executed: bool = False
    result_known: bool = True
    commit_eligible: bool = True


@dataclass
class ChunkSummary:
    skill_id: str | None = None
    planned_actions: int = 0
    accepted_actions: int = 0
    started_actions: int = 0
    executed_actions: int = 0
    completed_actions: int = 0
    failed_actions: int = 0
    rejected_actions: int = 0
    remaining_actions: int = 0
    terminal_status: str = "interrupted"
    stop_reason: str = ""
    explicit_dependency_check: str = "passed"
    observation_chars: int = 0
    chunk_id: str = field(default_factory=lambda: "chunk_" + uuid.uuid4().hex[:12])
    error: str = ""

    @property
    def chunk_length(self):
        return self.started_actions

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "skill_id": self.skill_id,
            "planned_actions": self.planned_actions,
            "accepted_actions": self.accepted_actions,
            "started_actions": self.started_actions,
            "executed_actions": self.executed_actions,
            "completed_actions": self.completed_actions,
            "failed_actions": self.failed_actions,
            "rejected_actions": self.rejected_actions,
            "remaining_actions": self.remaining_actions,
            "terminal_status": self.terminal_status,
            "stop_reason": self.stop_reason,
            "explicit_dependency_check": self.explicit_dependency_check,
            "observation_chars": self.observation_chars,
            "chunk_length": self.chunk_length,
            "error": self.error,
        }


def render_chunk_observation(summary):
    if summary.terminal_status == "completed":
        return ""
    if summary.terminal_status == "rejected":
        prefix = "Action chunk rejected."
    else:
        prefix = (
            f"Action chunk interrupted after {summary.started_actions}/"
            f"{summary.planned_actions} actions."
        )
    detail = summary.error or summary.stop_reason or "unknown boundary"
    return (
        "Runtime observation:\n"
        f"{prefix}\n"
        f"Reason: {detail}.\n"
        "Replan using independent read-only actions."
    )


@dataclass(frozen=True)
class ChunkExecution:
    summary: ChunkSummary
    results: tuple[ActionResult, ...] = ()


def _contains_result_reference(value):
    if isinstance(value, str):
        return bool(RESULT_REFERENCE_PATTERN.search(value))
    if isinstance(value, dict):
        return any(_contains_result_reference(key) or _contains_result_reference(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_result_reference(item) for item in value)
    return False


class ChunkValidator:
    def __init__(self, agent=None, allowed_tools=None, max_actions_per_chunk=4):
        self.agent = agent
        configured_tools = READ_ONLY_CHUNK_TOOLS if allowed_tools is None else allowed_tools
        self.allowed_tools = tuple(str(name).strip() for name in configured_tools)
        if not self.allowed_tools or any(not name for name in self.allowed_tools):
            raise ValueError("chunk validator requires non-empty tool names")
        invalid_tools = sorted(set(self.allowed_tools) - set(READ_ONLY_CHUNK_TOOLS))
        if invalid_tools:
            raise ValueError("chunk validator only supports read-only tools: " + ", ".join(invalid_tools))
        if isinstance(max_actions_per_chunk, bool):
            raise ValueError("max_actions_per_chunk must be positive")
        self.max_actions_per_chunk = int(max_actions_per_chunk)
        if self.max_actions_per_chunk < 1:
            raise ValueError("max_actions_per_chunk must be positive")

    def validate(self, payload, remaining_budget=None):
        try:
            chunk = ActionChunk.from_payload(payload)
        except ValueError as exc:
            planned = len(payload.get("actions", [])) if isinstance(payload, dict) and isinstance(payload.get("actions"), list) else 0
            return ChunkValidation(
                chunk=None,
                planned_actions=planned,
                rejected_actions=planned,
                error=str(exc),
            )

        planned = len(chunk.actions)
        if not planned:
            return ChunkValidation(
                chunk=chunk,
                planned_actions=0,
                error="chunk actions must not be empty",
            )
        if planned > self.max_actions_per_chunk:
            return ChunkValidation(
                chunk=chunk,
                planned_actions=planned,
                rejected_actions=planned,
                error=f"chunk contains {planned} actions; maximum is {self.max_actions_per_chunk}",
            )

        if remaining_budget is not None and int(remaining_budget) <= 0:
            return ChunkValidation(
                chunk=chunk,
                planned_actions=planned,
                accepted_actions=(),
                remaining_actions=planned,
                terminal_status="interrupted",
                stop_reason="budget_exhausted",
            )

        for action in chunk.actions:
            if _contains_result_reference(action.args):
                return ChunkValidation(
                    chunk=chunk,
                    planned_actions=planned,
                    rejected_actions=planned,
                    stop_reason="permission_or_policy_violation",
                    error="explicit action result references are not allowed",
                    explicit_dependency_check="failed",
                )
            if action.name not in self.allowed_tools:
                return ChunkValidation(
                    chunk=chunk,
                    planned_actions=planned,
                    rejected_actions=planned,
                    error=f"tool '{action.name}' is not allowed in action chunks",
                )
            if self.agent is not None:
                if self.agent.allowed_tools is not None and action.name not in self.agent.allowed_tools:
                    return ChunkValidation(
                        chunk=chunk,
                        planned_actions=planned,
                        rejected_actions=planned,
                        error=f"tool '{action.name}' is not allowed in this run",
                    )
                try:
                    self.agent.validate_tool(action.name, action.args)
                except Exception as exc:
                    return ChunkValidation(
                        chunk=chunk,
                        planned_actions=planned,
                        rejected_actions=planned,
                        error=f"invalid arguments for {action.name}: {exc}",
                    )

        if remaining_budget is None:
            accepted = chunk.actions
        else:
            remaining_budget = max(0, int(remaining_budget))
            accepted = chunk.actions[:remaining_budget]
        if not accepted:
            return ChunkValidation(
                chunk=chunk,
                planned_actions=planned,
                accepted_actions=(),
                remaining_actions=planned,
                terminal_status="interrupted",
                stop_reason="budget_exhausted",
            )
        return ChunkValidation(
            chunk=chunk,
            planned_actions=planned,
            accepted_actions=tuple(accepted),
            remaining_actions=planned - len(accepted),
            terminal_status="interrupted" if len(accepted) < planned else "completed",
            stop_reason="budget_exhausted" if len(accepted) < planned else "",
        )


class BoundaryPolicy:
    def __init__(self, observation_budget_chars=12000, skill_guidance_enabled=False):
        if isinstance(observation_budget_chars, bool) or int(observation_budget_chars) < 1:
            raise ValueError("observation_budget_chars must be positive")
        if not isinstance(skill_guidance_enabled, bool):
            raise ValueError("skill_guidance_enabled must be a boolean")
        self.observation_budget_chars = int(observation_budget_chars)
        self.skill_guidance_enabled = skill_guidance_enabled

    @staticmethod
    def select_stop_reason(reasons):
        reasons = set(reasons)
        for reason in STOP_REASON_PRIORITY:
            if reason in reasons:
                return reason
        return "hard_boundary"

    def after_action(
        self,
        result,
        observation_chars,
        *,
        boundary_hint=False,
        has_unstarted_actions=False,
        primitive_budget_exhausted=False,
    ):
        reasons = []
        if result.status == "failed":
            reasons.append("tool_failed")
        elif result.status == "rejected" or result.metadata.get("tool_status") == "rejected":
            reasons.append("permission_or_policy_violation")
        if result.metadata.get("process_crash_or_unknown_result"):
            reasons.append("process_crash_or_unknown_result")
        if result.metadata.get("truncated") or observation_chars > self.observation_budget_chars:
            reasons.append("observation_budget_exceeded")
        if result.metadata.get("hard_boundary"):
            reasons.append("hard_boundary")
        if self.skill_guidance_enabled and boundary_hint and result.status == "completed":
            reasons.append("skill_boundary")
        if primitive_budget_exhausted and has_unstarted_actions:
            reasons.append("budget_exhausted")
        return self.select_stop_reason(reasons) if reasons else ""


class PrimitiveActionRunner:
    """The only action submission path used by AgentLoop."""

    def __init__(self, agent, task_state, user_message):
        self.agent = agent
        self.task_state = task_state
        self.user_message = str(user_message)
        self.committer = ActionCommitter(agent)

    def run(self, action, remaining_budget=None):
        if remaining_budget is not None and int(remaining_budget) <= 0:
            return ActionResult(
                action_id="",
                action_seq=self.task_state.action_seq,
                action=action,
                status="interrupted",
                content="",
                metadata={"tool_error_code": "budget_exhausted"},
            )

        action_id = "action_" + uuid.uuid4().hex[:12]
        action_seq = self.task_state.next_action_seq()
        lifecycle = ActionState(action)
        lifecycle.transition("accepted")
        lifecycle.transition("started")
        self.agent.emit_trace(
            self.task_state,
            "action_started",
            {
                "action_id": action_id,
                "action_seq": action_seq,
                "name": action.name,
                "args": action.args,
            },
        )
        started_at = time.monotonic()
        try:
            tool_result = self.agent.execute_tool(action.name, action.args)
        except Exception as exc:
            tool_result = self.agent.tool_result_for_unknown_action(action.name, exc)
        metadata = dict(tool_result.metadata or {})
        tool_status = str(metadata.get("tool_status", "ok"))
        result_known_value = metadata.get("result_known")
        result_known = (
            bool(result_known_value)
            if result_known_value is not None
            else not bool(metadata.get("process_crash_or_unknown_result"))
        )
        if tool_status == "rejected":
            status = "rejected"
            executed = False
            lifecycle.transition("rejected")
        elif not result_known:
            status = "interrupted"
            executed = False
            lifecycle.transition("interrupted")
        else:
            status = "completed" if tool_status == "ok" else "failed"
            executed = True
            lifecycle.transition(status)
        # Every call that enters ToolExecutor consumes exactly one primitive
        # budget step, including a late pre-execution rejection.
        self.task_state.record_tool(action.name, status=status, executed=executed, result_known=result_known)
        result = ActionResult(
            action_id=action_id,
            action_seq=action_seq,
            action=action,
            status=status,
            content=tool_result.content,
            metadata=metadata,
            executed=executed,
            result_known=result_known,
            commit_eligible=result_known and bool(metadata.get("commit_eligible", True)),
        )
        if not result_known and not self.agent.tools.get(action.name, {}).get("risky", True):
            committed = checkpointlib.latest_committed_action(self.agent)
            self.task_state.action_seq = int(committed.get("action_seq", 0) or 0) if committed else 0
        checkpoint = self.committer.commit(self.task_state, result, self.user_message)
        self.agent.run_store.write_task_state(self.task_state)
        self.agent.emit_trace(
            self.task_state,
            "tool_executed",
            {
                "action_id": action_id,
                "action_seq": action_seq,
                "action_status": status,
                "name": action.name,
                "args": action.args,
                "result": clip(result.content, 500),
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                **metadata,
            },
        )
        if checkpoint:
            self.agent.emit_trace(
                self.task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "action_committed",
                    "action_id": action_id,
                    "action_seq": action_seq,
                    "action_status": status,
                },
            )
        return ActionResult(
            action_id=result.action_id,
            action_seq=result.action_seq,
            action=result.action,
            status=result.status,
            content=result.content,
            metadata=result.metadata,
            checkpoint_id=checkpoint["checkpoint_id"] if checkpoint else "",
            executed=result.executed,
            result_known=result.result_known,
            commit_eligible=result.commit_eligible,
        )


class ChunkExecutor:
    def __init__(self, runner, validator, boundary_policy):
        self.runner = runner
        self.validator = validator
        self.boundary_policy = boundary_policy

    def run(self, payload, remaining_budget):
        validation = self.validator.validate(payload, remaining_budget=remaining_budget)
        if validation.chunk is None or validation.rejected_actions or validation.error:
            summary = ChunkSummary(
                chunk_id=(
                    validation.chunk.chunk_id
                    if validation.chunk and validation.chunk.chunk_id
                    else "chunk_" + uuid.uuid4().hex[:12]
                ),
                planned_actions=validation.planned_actions,
                accepted_actions=len(validation.accepted_actions),
                rejected_actions=validation.rejected_actions or validation.planned_actions,
                remaining_actions=validation.remaining_actions,
                terminal_status="rejected",
                stop_reason=validation.stop_reason,
                explicit_dependency_check=validation.explicit_dependency_check,
                error=validation.error,
            )
            return ChunkExecution(summary=summary)

        chunk = validation.chunk
        summary = ChunkSummary(
            chunk_id=chunk.chunk_id or ("chunk_" + uuid.uuid4().hex[:12]),
            skill_id=chunk.skill_id,
            planned_actions=validation.planned_actions,
            accepted_actions=len(validation.accepted_actions),
            remaining_actions=validation.remaining_actions,
            explicit_dependency_check=validation.explicit_dependency_check,
        )
        results = []
        observation_chars = 0
        for index, action in enumerate(validation.accepted_actions):
            # A submitted primitive consumes one step even when ToolExecutor
            # returns a late rejection.  Use started_actions for the budget;
            # executed_actions intentionally measures successful tool entry.
            budget_remaining = int(remaining_budget) - summary.started_actions
            if budget_remaining <= 0:
                summary.stop_reason = "budget_exhausted"
                break
            result = self.runner.run(action, remaining_budget=budget_remaining)
            results.append(result)
            summary.started_actions += 1
            summary.executed_actions += int(result.executed)
            summary.completed_actions += int(result.status == "completed")
            summary.failed_actions += int(result.status == "failed")
            summary.rejected_actions += int(result.status == "rejected")
            observation_chars += len(result.content)
            summary.observation_chars = observation_chars
            is_last_accepted = index == len(validation.accepted_actions) - 1
            stop_reason = self.boundary_policy.after_action(
                result,
                observation_chars,
                boundary_hint=action.boundary_hint or (chunk.boundary_hint and is_last_accepted),
                has_unstarted_actions=(
                    not is_last_accepted or validation.remaining_actions > 0
                ),
                primitive_budget_exhausted=(
                    summary.started_actions >= int(remaining_budget)
                ),
            )
            if stop_reason:
                summary.stop_reason = stop_reason
                break

        summary.remaining_actions = max(0, summary.planned_actions - summary.started_actions)
        if summary.stop_reason:
            summary.terminal_status = "interrupted"
        elif summary.started_actions == summary.planned_actions:
            summary.terminal_status = "completed"
            summary.stop_reason = "chunk_completed"
        else:
            summary.terminal_status = "interrupted"
            summary.stop_reason = "budget_exhausted"
        return ChunkExecution(summary=summary, results=tuple(results))
