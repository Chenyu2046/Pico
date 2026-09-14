import copy
import json

import pytest

from pico import FakeModelClient, Pico, SessionStore, WorkspaceContext
from pico import checkpoint as checkpointlib
from pico.checkpoint import latest_committed_action
from pico.tool_executor import ToolExecutionResult


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    return Pico(
        model_client=FakeModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def tool(path):
    return f'<tool>{{"name":"read_file","args":{{"path":"{path}","start":1,"end":1}}}}</tool>'


def chunk(*actions):
    return "<chunk>" + json.dumps({"actions": list(actions)}) + "</chunk>"


def read_action(path):
    return {"name": "read_file", "args": {"path": path, "start": 1, "end": 1}}


def action_checkpoints(session):
    return [
        item
        for item in session["checkpoints"]["items"].values()
        if item.get("action_id")
    ]


def trace_events(agent):
    return [
        json.loads(line)
        for line in agent.run_store.trace_path(agent.current_task_state)
        .read_text(encoding="utf-8")
        .splitlines()
    ]


class FailBeforeReplaceOnActionStore(SessionStore):
    def __init__(self, root):
        super().__init__(root)
        self.fail_action_commit = False
        self.successful_snapshots = []

    def save(self, session):
        has_action_history = any(
            item.get("action_id") for item in session.get("history", [])
        )
        if self.fail_action_commit and has_action_history:
            raise RuntimeError("injected before durable replace")
        path = super().save(session)
        self.successful_snapshots.append(copy.deepcopy(self.load(session["id"])))
        return path


def test_failed_action_commit_does_not_leak_dirty_state_on_later_record(tmp_path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    store = FailBeforeReplaceOnActionStore(tmp_path / ".pico" / "sessions")
    agent = Pico(
        model_client=FakeModelClient([tool("a.txt")]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )
    store.fail_action_commit = True

    with pytest.raises(RuntimeError, match="before durable replace"):
        agent.ask("inspect")

    durable_before_failure = store.successful_snapshots[-1]
    store.fail_action_commit = False
    later_record = {
        "role": "assistant",
        "content": "later durable note",
        "created_at": "2026-09-15T00:00:00+00:00",
    }
    agent.record(later_record)

    recovered = store.load(agent.session["id"])
    assert recovered["history"] == durable_before_failure["history"] + [later_record]
    assert recovered["memory"] == durable_before_failure["memory"]
    assert recovered["checkpoints"] == durable_before_failure["checkpoints"]
    assert latest_committed_action(agent) is None
    assert not any(item.get("action_id") for item in recovered["history"])

    task_state = json.loads(
        agent.run_store.task_state_path(agent.current_task_state).read_text(encoding="utf-8")
    )
    assert task_state["checkpoint_id"] == ""
    assert task_state["action_seq"] == 0


def test_recovery_discards_legacy_checkpoint_tail_and_reverts_current_id(tmp_path):
    agent = build_agent(
        tmp_path,
        [tool("a.txt"), "<final>done</final>"],
    )
    assert agent.ask("inspect") == "done"

    session = agent.session_store.load(agent.session["id"])
    committed = latest_committed_action(agent)
    assert committed["action_seq"] == 1
    tail_id = "ckpt_uncommitted_seq2"
    non_action_id = "ckpt_non_action_seq2"
    non_action = copy.deepcopy(committed)
    non_action.update(
        {
            "checkpoint_id": non_action_id,
            "parent_checkpoint_id": committed["checkpoint_id"],
            "current_goal": "non-action checkpoint must survive",
            "action_id": "",
            "action_seq": 2,
            "committed": False,
            "commit_status": "not-applicable",
            "action_status": "",
            "created_at": "2026-09-14T00:00:00+00:00",
        }
    )
    session["checkpoints"]["items"][non_action_id] = non_action
    tail = copy.deepcopy(committed)
    tail.update(
        {
            "checkpoint_id": tail_id,
            "parent_checkpoint_id": committed["checkpoint_id"],
            "current_goal": "seq2 tail must be discarded",
            "action_id": "action_uncommitted_seq2",
            "action_seq": 2,
            "committed": False,
            "commit_status": "not-committed",
            "action_status": "completed",
        }
    )
    session["checkpoints"]["items"][tail_id] = tail
    session["checkpoints"]["current_id"] = tail_id
    session["history"].append(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "b.txt", "start": 1, "end": 1},
            "content": "stale seq2 tail",
            "action_id": "action_uncommitted_seq2",
            "action_seq": 2,
            "action_status": "completed",
        }
    )
    agent.session_store.save(session)

    recovered = Pico.from_session(
        FakeModelClient(["<final>resumed</final>"]),
        agent.workspace,
        agent.session_store,
        agent.session["id"],
        approval_policy="auto",
    )

    assert recovered.resume_state["recovery_tail_discarded"] == 1
    assert recovered.resume_state["last_committed_action"]["action_seq"] == 1
    assert latest_committed_action(recovered)["action_seq"] == 1
    assert recovered.session["checkpoints"]["current_id"] == committed["checkpoint_id"]
    assert tail_id not in recovered.session["checkpoints"]["items"]
    assert non_action_id in recovered.session["checkpoints"]["items"]
    assert not any(item.get("action_seq") == 2 for item in recovered.session["history"])
    assert "seq2 tail must be discarded" not in recovered.render_checkpoint_text()

    assert recovered.ask("resume") == "resumed"
    assert "seq2 tail must be discarded" not in recovered.model_client.prompts[-1]


def test_action_sequence_mismatch_fails_closed_before_followup_action_or_planning(
    tmp_path, monkeypatch
):
    agent = build_agent(
        tmp_path,
        [
            chunk(read_action("a.txt"), read_action("b.txt")),
            "<final>must not be consumed</final>",
        ],
        action_chunking={"enabled": True},
        max_steps=3,
    )
    real_latest = checkpointlib.latest_committed_action
    mismatch_active = False
    executed_paths = []

    def mismatched_latest(target):
        if mismatch_active and target is agent:
            return {
                "checkpoint_id": "ckpt_foreign_watermark",
                "action_id": "action_foreign",
                "action_seq": 99,
                "committed": True,
                "commit_status": "committed",
                "schema_version": "phase1-v1",
            }
        return real_latest(target)

    monkeypatch.setattr(checkpointlib, "latest_committed_action", mismatched_latest)
    original_execute = agent.execute_tool

    def activate_mismatch(name, args):
        nonlocal mismatch_active
        mismatch_active = True
        executed_paths.append(args["path"])
        return original_execute(name, args)

    agent.execute_tool = activate_mismatch
    assert agent.ask("inspect")
    mismatch_active = False

    assert agent.current_task_state.stop_reason == "action_sequence_mismatch"
    assert agent.current_task_state.status == "stopped"
    assert agent.current_task_state.last_chunk["stop_reason"] == "action_sequence_mismatch"
    assert executed_paths == ["a.txt"]
    assert len(agent.model_client.prompts) == 1
    assert latest_committed_action(agent) is None
    assert not action_checkpoints(agent.session)
    assert not any(item.get("action_id") for item in agent.session["history"])

    events = trace_events(agent)
    assert any(
        event["event"] == "action_sequence_mismatch"
        and event["action_seq"] == 1
        and event["watermark"] == 99
        for event in events
    )


def test_failed_read_file_commits_result_and_process_note_but_not_error_memory(
    tmp_path,
):
    agent = build_agent(
        tmp_path,
        [tool("a.txt"), "<final>replanned</final>"],
    )
    agent.execute_tool = lambda name, args: ToolExecutionResult(
        "error: injected read failure",
        {
            "tool_status": "error",
            "tool_error_code": "tool_failed",
            "affected_paths": ["a.txt"],
        },
    )

    assert agent.ask("inspect") == "replanned"

    action = latest_committed_action(agent)
    assert action["action_seq"] == 1
    assert action["action_status"] == "failed"
    tool_history = [item for item in agent.session["history"] if item.get("role") == "tool"]
    assert tool_history and tool_history[0]["content"] == "error: injected read failure"

    memory = agent.memory.to_dict()
    assert "a.txt" not in memory["file_summaries"]
    trusted_notes = [
        note for note in memory["episodic_notes"] if note.get("kind") != "process"
    ]
    assert not trusted_notes
    process_notes = [
        note for note in memory["episodic_notes"] if note.get("kind") == "process"
    ]
    assert process_notes
    assert process_notes[-1]["text"] == "read_file error on a.txt; check the failure before retry"
    assert "injected read failure" not in json.dumps(trusted_notes)


def _seed_uncommitted_tail(tmp_path):
    agent = build_agent(tmp_path, [tool("a.txt"), "<final>done</final>"])
    assert agent.ask("seed") == "done"
    session = agent.session_store.load(agent.session["id"])
    committed = latest_committed_action(agent)
    tail_id = "ckpt_recovery_tail"
    tail = copy.deepcopy(committed)
    tail.update(
        {
            "checkpoint_id": tail_id,
            "parent_checkpoint_id": committed["checkpoint_id"],
            "action_id": "action_recovery_tail",
            "action_seq": 2,
            "committed": False,
            "commit_status": "not-committed",
        }
    )
    session["checkpoints"]["items"][tail_id] = tail
    session["checkpoints"]["current_id"] = tail_id
    session["history"].append(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "b.txt"},
            "content": "uncommitted recovery tail",
            "action_id": "action_recovery_tail",
            "action_seq": 2,
        }
    )
    agent.session_store.save(session)
    return Pico.from_session(
        FakeModelClient([]),
        agent.workspace,
        agent.session_store,
        agent.session["id"],
        approval_policy="auto",
    )


def test_recovery_discard_trace_is_once_per_run_with_new_and_cumulative_counts(tmp_path):
    agent = _seed_uncommitted_tail(tmp_path)
    agent.model_client.outputs = ['<tool>{"args":{}}</tool>', "<final>first</final>"]
    assert agent.ask("first resume") == "first"
    first_events = [
        event for event in trace_events(agent) if event["event"] == "recovery_tail_discarded"
    ]

    agent.model_client.outputs = ['<tool>{"args":{}}</tool>', "<final>second</final>"]
    assert agent.ask("second resume") == "second"
    second_events = [
        event for event in trace_events(agent) if event["event"] == "recovery_tail_discarded"
    ]

    assert len(first_events) == 1
    assert first_events[0]["new_count"] == 1
    assert first_events[0]["cumulative_count"] == 1
    assert len(second_events) == 1
    assert second_events[0]["new_count"] == 0
    assert second_events[0]["cumulative_count"] == 1
