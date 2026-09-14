"""Atomic durable commit for one primitive action result."""

from . import checkpoint as checkpointlib
from .workspace import now


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
            return None
        if action_result.executed:
            agent.update_memory_after_tool(
                action_result.action.name,
                action_result.action.args,
                action_result.content,
            )
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
