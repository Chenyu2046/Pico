"""一次 ask() 运行过程中的状态机快照。

它回答的是：这次用户请求当前进行到哪了、调了多少次工具、最后为什么停下。
这个对象会被不断写入 task_state.json，供运行中观察和运行后复盘。
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_TOOL_TIMEOUT = "tool_timeout"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_DELEGATE_FAILED = "delegate_failed"
STOP_REASON_PERSISTENCE_ERROR = "persistence_error"
STOP_REASON_RESUME_LOAD_ERROR = "resume_load_error"
STOP_REASON_UNKNOWN_RESULT = "unknown_tool_result"
STOP_REASON_ACTION_SEQUENCE_MISMATCH = "action_sequence_mismatch"


@dataclass
class TaskState:
    run_id: str
    task_id: str
    user_request: str
    status: str = STATUS_RUNNING
    tool_steps: int = 0
    attempts: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    checkpoint_id: str = ""
    resume_status: str = ""
    logical_decisions: int = 0
    provider_requests: int = 0
    provider_retries: int = 0
    provider_responses: int = 0
    provider_attempts: list = field(default_factory=list)
    usage_missing_responses: int = 0
    auxiliary_requests: int = 0
    primitive_tool_calls: int = 0
    primitive_submissions: int = 0
    executed_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    rejected_tool_calls: int = 0
    unknown_tool_calls: int = 0
    action_seq: int = 0
    chunk_count: int = 0
    chunk_interrupts: int = 0
    chunk_lengths: list = field(default_factory=list)
    last_chunk: dict = field(default_factory=dict)

    @classmethod
    def create(cls, task_id, user_request, run_id=""):
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, data):
        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            user_request=str(data.get("user_request", "")),
            status=str(data.get("status", STATUS_RUNNING)),
            tool_steps=int(data.get("tool_steps", 0)),
            attempts=int(data.get("attempts", 0)),
            last_tool=str(data.get("last_tool", "")),
            stop_reason=str(data.get("stop_reason", "")),
            final_answer=str(data.get("final_answer", "")),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            resume_status=str(data.get("resume_status", "")),
            logical_decisions=int(data.get("logical_decisions", 0)),
            provider_requests=int(data.get("provider_requests", 0)),
            provider_retries=int(data.get("provider_retries", 0)),
            provider_responses=int(data.get("provider_responses", 0)),
            provider_attempts=list(data.get("provider_attempts", [])),
            usage_missing_responses=int(data.get("usage_missing_responses", 0)),
            auxiliary_requests=int(data.get("auxiliary_requests", 0)),
            primitive_tool_calls=int(data.get("primitive_tool_calls", data.get("tool_steps", 0))),
            primitive_submissions=int(data.get("primitive_submissions", data.get("tool_steps", 0))),
            executed_tool_calls=int(data.get("executed_tool_calls", 0)),
            successful_tool_calls=int(data.get("successful_tool_calls", 0)),
            failed_tool_calls=int(data.get("failed_tool_calls", 0)),
            rejected_tool_calls=int(data.get("rejected_tool_calls", 0)),
            unknown_tool_calls=int(data.get("unknown_tool_calls", 0)),
            action_seq=int(data.get("action_seq", 0)),
            chunk_count=int(data.get("chunk_count", 0)),
            chunk_interrupts=int(data.get("chunk_interrupts", 0)),
            chunk_lengths=list(data.get("chunk_lengths", [])),
            last_chunk=dict(data.get("last_chunk", {}) or {}),
        )

    def record_attempt(self):
        # attempt 统计的是“模型被调用了几轮”，不等于 tool_steps。
        self.attempts += 1
        return self

    def record_logical_decision(self):
        self.logical_decisions += 1
        return self

    def record_provider_requests(self, requests=1, retries=0, usage_missing=False, attempts=None):
        self.provider_requests += max(0, int(requests))
        self.provider_retries += max(0, int(retries))
        self.provider_responses += 1
        self.usage_missing_responses += int(bool(usage_missing))
        if attempts:
            self.provider_attempts.extend(dict(attempt) for attempt in attempts)
        return self

    def record_tool(self, name, status="completed", executed=True, result_known=True):
        # tool_steps 统计 primitive submission，而不是成功执行次数。
        self.tool_steps += 1
        self.primitive_tool_calls += 1
        self.primitive_submissions += 1
        if executed:
            self.executed_tool_calls += 1
        if not result_known:
            self.unknown_tool_calls += 1
        elif status == "completed":
            self.successful_tool_calls += 1
        elif status == "failed":
            self.failed_tool_calls += 1
        elif status == "rejected":
            self.rejected_tool_calls += 1
        self.last_tool = str(name or "")
        return self

    def next_action_seq(self):
        self.action_seq += 1
        return self.action_seq

    def record_chunk(self, summary):
        payload = dict(summary.to_dict() if hasattr(summary, "to_dict") else summary)
        self.chunk_count += 1
        self.chunk_lengths.append(int(payload.get("chunk_length", payload.get("started_actions", 0))))
        if payload.get("terminal_status") != "completed":
            self.chunk_interrupts += 1
        self.last_chunk = payload
        return self

    def stop(self, stop_reason, status=STATUS_STOPPED, final_answer=""):
        # stop_reason 和 status 分开存，是为了区分“怎么停的”和“停下时是什么状态”。
        self.status = status
        self.stop_reason = stop_reason
        if final_answer != "":
            self.final_answer = final_answer
        return self

    def stop_step_limit(self, final_answer=""):
        return self.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer=final_answer)

    def stop_retry_limit(self, final_answer=""):
        return self.stop(STOP_REASON_RETRY_LIMIT_REACHED, final_answer=final_answer)

    def stop_model_error(self, final_answer=""):
        return self.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED, final_answer=final_answer)

    def finish_success(self, final_answer):
        self.status = STATUS_COMPLETED
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)
        return self

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
            "logical_decisions": self.logical_decisions,
            "provider_requests": self.provider_requests,
            "provider_retries": self.provider_retries,
            "provider_responses": self.provider_responses,
            "provider_attempts": [dict(attempt) for attempt in self.provider_attempts],
            "usage_missing_responses": self.usage_missing_responses,
            "auxiliary_requests": self.auxiliary_requests,
            "primitive_tool_calls": self.primitive_tool_calls,
            "primitive_submissions": self.primitive_submissions,
            "executed_tool_calls": self.executed_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "rejected_tool_calls": self.rejected_tool_calls,
            "unknown_tool_calls": self.unknown_tool_calls,
            "action_seq": self.action_seq,
            "chunk_count": self.chunk_count,
            "chunk_interrupts": self.chunk_interrupts,
            "chunk_lengths": list(self.chunk_lengths),
            "last_chunk": dict(self.last_chunk),
        }
