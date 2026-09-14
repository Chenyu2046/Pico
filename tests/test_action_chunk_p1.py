import copy
import json

import pytest

from pico import FakeModelClient, Pico, SessionStore, WorkspaceContext
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


def chunk(*actions):
    return "<chunk>" + json.dumps({"actions": list(actions)}) + "</chunk>"


def read_action(path):
    return {"name": "read_file", "args": {"path": path, "start": 1, "end": 1}}


def action_checkpoints(agent):
    return [
        item
        for item in agent.session["checkpoints"]["items"].values()
        if item.get("action_id")
    ]


def test_exact_fit_chunk_is_completed(tmp_path):
    agent = build_agent(
        tmp_path,
        [chunk(read_action("a.txt"), read_action("b.txt"), read_action("README.md")), "<final>done</final>"],
        action_chunking={"enabled": True},
        max_steps=3,
    )

    assert agent.ask("inspect") == "done"
    assert agent.current_task_state.last_chunk["terminal_status"] == "completed"
    assert agent.current_task_state.last_chunk["stop_reason"] == "chunk_completed"


def test_chunk_over_budget_stops_after_primitive_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [chunk(read_action("a.txt"), read_action("b.txt"), read_action("README.md"), read_action("a.txt")), "<final>done</final>"],
        action_chunking={"enabled": True},
        max_steps=3,
    )

    assert agent.ask("inspect") == "done"
    summary = agent.current_task_state.last_chunk
    assert agent.current_task_state.tool_steps == 3
    assert summary["started_actions"] == 3
    assert summary["terminal_status"] == "interrupted"
    assert summary["stop_reason"] == "budget_exhausted"


def test_failed_action_stops_the_rest_of_the_chunk(tmp_path):
    agent = build_agent(
        tmp_path,
        [chunk(read_action("a.txt"), read_action("b.txt"), read_action("a.txt")), "<final>done</final>"],
        action_chunking={"enabled": True},
        max_steps=3,
    )
    original_execute = agent.execute_tool

    def fail_second_action(name, args):
        if args.get("path") == "b.txt":
            return ToolExecutionResult("error: injected tool failure", {"tool_status": "error"})
        return original_execute(name, args)

    agent.execute_tool = fail_second_action

    assert agent.ask("inspect") == "done"
    assert agent.current_task_state.tool_steps == 2
    assert len([item for item in agent.session["history"] if item["role"] == "tool"]) == 2


def test_invalid_chunk_submits_no_primitive_action(tmp_path):
    agent = build_agent(
        tmp_path,
        [chunk({"name": "write_file", "args": {"path": "x", "content": "x"}}), "<final>done</final>"],
        action_chunking={"enabled": True},
    )

    assert agent.ask("inspect") == "done"
    assert agent.current_task_state.tool_steps == 0
    assert not any(item["role"] == "tool" for item in agent.session["history"])


def test_empty_chunk_is_rejected_and_explained(tmp_path):
    agent = build_agent(
        tmp_path,
        [chunk(), "<final>replanned</final>"],
        action_chunking={"enabled": True},
    )

    assert agent.ask("inspect") == "replanned"
    summary = agent.current_task_state.last_chunk
    assert summary["terminal_status"] == "rejected"
    assert summary["stop_reason"] == "permission_or_policy_violation"
    assert "chunk actions must not be empty" in agent.model_client.prompts[1]
    assert agent.current_task_state.tool_steps == 0


def test_rejected_chunk_reason_is_visible_to_next_prompt(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            chunk({"name": "read_file", "args": {"path": "${action_1.result}"}}),
            "<final>replanned</final>",
        ],
        action_chunking={"enabled": True},
    )

    assert agent.ask("inspect") == "replanned"
    assert "explicit action result references are not allowed" in agent.model_client.prompts[1]


def test_interrupted_chunk_reason_is_visible_to_next_prompt(tmp_path):
    agent = build_agent(
        tmp_path,
        [chunk(read_action("a.txt"), read_action("b.txt")), "<final>replanned</final>"],
        action_chunking={"enabled": True, "observation_budget_chars": 1},
    )

    assert agent.ask("inspect") == "replanned"
    assert "observation_budget_exceeded" in agent.model_client.prompts[1]


def test_disabled_chunk_notice_does_not_advertise_chunk_protocol(tmp_path):
    agent = build_agent(tmp_path, [chunk(read_action("a.txt")), "<final>done</final>"])

    assert agent.ask("inspect") == "done"
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert notices
    assert "<chunk>" not in notices[0]


def test_retry_notice_advertises_only_enabled_protocols():
    kind, enabled_notice = Pico.parse(
        '<tool>{"args":{}}</tool>',
        action_chunking={"enabled": True},
    )
    assert kind == "retry"
    assert "valid <chunk> call" in enabled_notice

    kind, disabled_notice = Pico.parse('<tool>{"args":{}}</tool>')
    assert kind == "retry"
    assert "valid <chunk> call" not in disabled_notice


def test_successful_action_advances_committed_watermark(tmp_path):
    agent = build_agent(
        tmp_path,
        ['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>', "<final>done</final>"],
    )

    assert agent.ask("inspect") == "done"
    checkpoint = action_checkpoints(agent)[0]
    assert checkpoint["action_seq"] == 1
    assert checkpoint["action_status"] == "completed"
    assert checkpoint["result_known"] is True
    assert checkpoint["committed"] is True
    assert latest_committed_action(agent)["action_seq"] == 1


def test_known_failed_action_advances_committed_watermark(tmp_path):
    agent = build_agent(
        tmp_path,
        ['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>', "<final>done</final>"],
    )

    def fail_after_execution(name, args):
        return ToolExecutionResult("error: injected tool failure", {"tool_status": "error"})

    agent.execute_tool = fail_after_execution

    assert agent.ask("inspect") == "done"
    checkpoint = action_checkpoints(agent)[0]
    assert checkpoint["action_status"] == "failed"
    assert checkpoint["result_known"] is True
    assert checkpoint["committed"] is True
    assert latest_committed_action(agent)["action_seq"] == 1


def test_known_rejected_action_advances_committed_watermark(tmp_path):
    agent = build_agent(
        tmp_path,
        ['<tool>{"name":"search","args":{"pattern":"a"}}</tool>', "<final>done</final>"],
        allowed_tools=["read_file"],
    )

    assert agent.ask("inspect") == "done"
    checkpoint = action_checkpoints(agent)[0]
    assert checkpoint["action_status"] == "rejected"
    assert checkpoint["result_known"] is True
    assert checkpoint["committed"] is True
    assert latest_committed_action(agent)["action_seq"] == 1


def test_unknown_action_result_does_not_commit_or_advance_watermark(tmp_path):
    agent = build_agent(
        tmp_path,
        ['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>', "<final>done</final>"],
    )

    def unknown_result(name, args):
        raise RuntimeError("simulated executor crash")

    agent.execute_tool = unknown_result

    assert agent.ask("inspect") == "done"
    assert latest_committed_action(agent) is None
    assert not action_checkpoints(agent)
    assert not any(item.get("role") == "tool" for item in agent.session["history"])
    assert agent.current_task_state.tool_steps == 1
    assert agent.current_task_state.unknown_tool_calls == 1
    assert agent.current_task_state.executed_tool_calls == 0


def test_explicit_unknown_result_metadata_does_not_commit(tmp_path):
    agent = build_agent(
        tmp_path,
        ['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>', "<final>done</final>"],
    )

    agent.execute_tool = lambda name, args: ToolExecutionResult(
        "executor result unavailable",
        {"tool_status": "ok", "result_known": False},
    )

    assert agent.ask("inspect") == "done"
    assert latest_committed_action(agent) is None
    assert not action_checkpoints(agent)
    assert agent.current_task_state.unknown_tool_calls == 1


def test_unknown_read_only_result_can_replan_from_committed_watermark(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"b.txt","start":1,"end":1}}</tool>',
            "<final>done</final>",
        ],
    )
    calls = []
    original_execute = agent.execute_tool

    def unknown_once(name, args):
        calls.append(args["path"])
        if len(calls) == 1:
            raise RuntimeError("simulated read crash")
        return original_execute(name, args)

    agent.execute_tool = unknown_once

    assert agent.ask("inspect") == "done"
    committed = latest_committed_action(agent)
    assert committed["action_seq"] == 1
    committed_tools = [item for item in agent.session["history"] if item.get("role") == "tool"]
    assert [item["args"]["path"] for item in committed_tools] == ["b.txt"]
    assert agent.current_task_state.unknown_tool_calls == 1


def test_tool_executor_has_no_memory_commit_side_effect(tmp_path):
    agent = build_agent(tmp_path, [])
    before = copy.deepcopy(agent.session["memory"])

    agent.execute_tool("read_file", {"path": "a.txt", "start": 1, "end": 1})

    assert agent.session["memory"] == before


class FailBeforeCommitStore(SessionStore):
    def __init__(self, root):
        super().__init__(root)
        self.save_count = 0

    def save(self, session):
        self.save_count += 1
        if getattr(self, "fail", False) and self.save_count >= 3:
            raise RuntimeError("injected before session commit")
        return super().save(session)


class FailAfterCommitStore(SessionStore):
    def __init__(self, root):
        super().__init__(root)
        self.save_count = 0

    def save(self, session):
        self.save_count += 1
        path = super().save(session)
        if getattr(self, "fail", False) and self.save_count >= 3:
            raise RuntimeError("injected after session commit")
        return path


def test_crash_before_session_commit_does_not_advance_watermark(tmp_path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    store = FailBeforeCommitStore(tmp_path / ".pico" / "sessions")
    agent = Pico(
        model_client=FakeModelClient(['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>']),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )
    store.fail = True

    with pytest.raises(RuntimeError, match="before session commit"):
        agent.ask("inspect")

    recovered = store.load(agent.session["id"])
    assert not recovered["checkpoints"]["items"]


def test_crash_after_session_commit_keeps_watermark(tmp_path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    store = FailAfterCommitStore(tmp_path / ".pico" / "sessions")
    agent = Pico(
        model_client=FakeModelClient(['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>']),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )
    store.fail = True

    with pytest.raises(RuntimeError, match="after session commit"):
        agent.ask("inspect")

    recovered = store.load(agent.session["id"])
    committed = [
        item for item in recovered["checkpoints"]["items"].values() if item.get("committed")
    ]
    assert committed
    assert committed[-1]["action_seq"] == 1


def test_recovery_discards_action_tail_above_committed_watermark(tmp_path):
    agent = build_agent(
        tmp_path,
        ['<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":1}}</tool>', "<final>done</final>"],
    )
    assert agent.ask("inspect") == "done"
    session = agent.session_store.load(agent.session["id"])
    session["history"].append(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "b.txt"},
            "content": "stale tail",
            "action_id": "action_tail",
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

    assert not any(item.get("action_seq") == 2 for item in recovered.session["history"])
    assert recovered.session["resume_state"]["recovery_tail_discarded"] == 1
