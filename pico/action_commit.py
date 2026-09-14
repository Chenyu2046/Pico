"""Atomic durable commit for one primitive action result."""

import copy

from . import checkpoint as checkpointlib
from .features.memory import LayeredMemory
from .workspace import now


class ActionCommitSequenceError(RuntimeError):
    """The next action is not the one immediately after the durable watermark."""

    def __init__(self, action_seq, watermark, action_result=None):
        self.action_seq = int(action_seq)
        self.watermark = int(watermark)
        self.action_result = action_result
        super().__init__(
            f"action commit sequence mismatch: action_seq={self.action_seq}, "
            f"expected={self.watermark + 1}, watermark={self.watermark}"
        )


def _durable_action_present(agent, action_result):
    try:
        persisted = agent.session_store.load(agent.session["id"])
    except Exception:
        return False
    items = persisted.get("checkpoints", {}).get("items", {})
    return any(
        item.get("action_id") == action_result.action_id
        and int(item.get("action_seq", 0) or 0) == int(action_result.action_seq)
        and checkpointlib._is_complete_committed_action(item)
        for item in items.values()
        if isinstance(item, dict)
    )


class ActionCommitter:
    """Apply one known action result and save the session exactly once."""

    def __init__(self, agent):
        self.agent = agent

    def commit(self, task_state, action_result, user_message):
        if not action_result.result_known or not action_result.commit_eligible:
            return None

        agent = self.agent
        committed = checkpointlib.latest_committed_action(agent)
        watermark = int(committed.get("action_seq", 0) or 0) if committed else 0
        if action_result.action_seq != watermark + 1:
            raise ActionCommitSequenceError(action_result.action_seq, watermark, action_result)

        session_before = copy.deepcopy(agent.session)
        checkpoint_id_before = task_state.checkpoint_id
        try:
            if action_result.executed and action_result.status == "completed":
                agent.update_memory_after_tool(
                    action_result.action.name,
                    action_result.action.args,
                    action_result.content,
                )
            if action_result.status in {"failed", "rejected"}:
                agent.record_process_note_for_tool(
                    action_result.action.name,
                    action_result.metadata,
                )
            agent.session["memory"] = agent.memory.to_dict()
            agent.record(
                {
                    "role": "tool",
                    "name": action_result.action.name,
                    "args": action_result.action.args,
                    "content": action_result.content,
                    "created_at": now(),
                    "action_id": action_result.action_id,
                    "action_seq": action_result.action_seq,
                    "action_status": action_result.status,
                },
                persist=False,
            )
            checkpoint = checkpointlib.build_checkpoint(
                agent,
                task_state,
                user_message,
                trigger="action_committed",
                action_result=action_result,
            )
            checkpointlib.attach_checkpoint(agent, task_state, checkpoint)
            agent.session_path = agent.session_store.save(agent.session)
            return checkpoint
        except Exception:
            # A post-replace exception means the durable commit exists and the
            # live state already matches it. Before replace, restore all live
            # state so a later save cannot publish an uncommitted action.
            if not _durable_action_present(agent, action_result):
                agent.session = session_before
                agent.memory = LayeredMemory(
                    agent.session["memory"],
                    workspace_root=agent.root,
                )
                agent.resume_state = agent.session.get("resume_state", {})
                task_state.checkpoint_id = checkpoint_id_before
                task_state.action_seq = watermark
            raise
