"""Agent control loop extracted from the runtime facade."""

import inspect
import time

from .action_chunk import (
    Action,
    BoundaryPolicy,
    ChunkExecutor,
    ChunkValidator,
    PrimitiveActionRunner,
    render_chunk_observation,
)
from .checkpoint import (
    CHECKPOINT_NONE_STATUS,
    CHECKPOINT_PARTIAL_STALE_STATUS,
    CHECKPOINT_WORKSPACE_MISMATCH_STATUS,
    latest_committed_action,
)
from .task_state import STOP_REASON_UNKNOWN_RESULT, TaskState
from .workspace import clip, now


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def _persist_model_failure(self, task_state, user_message, exc, run_started_at, prompt_metadata):
        agent = self.agent
        error_text = agent.redact_text(str(exc))
        final = f"Model request failed: {error_text}"
        task_state.stop_model_error(final)
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger="model_error")
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "model_failed",
            {
                "error": error_text,
                "completion_metadata": dict(agent.last_completion_metadata),
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.last_prompt_metadata = dict(prompt_metadata)
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))

    def _request_model(
        self,
        task_state,
        user_message,
        prompt,
        prompt_metadata,
        run_started_at,
        purpose,
        logical_decision_id,
    ):
        agent = self.agent
        agent.emit_trace(
            task_state,
            "model_requested",
            {
                "attempts": task_state.attempts,
                "tool_steps": task_state.tool_steps,
                "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                "purpose": purpose,
                "logical_decision_id": logical_decision_id,
            },
        )
        prompt_cache_key = None
        prompt_cache_retention = None
        if getattr(agent.model_client, "supports_prompt_cache", False):
            prompt_cache_key = prompt_metadata.get("prompt_cache_key")
            prompt_cache_retention = "in_memory"
        if hasattr(agent.model_client, "last_provider_metadata"):
            agent.model_client.last_provider_metadata = {}
        model_started_at = time.monotonic()
        complete_kwargs = {
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
        }
        if self._complete_accepts_logical_decision_id(agent.model_client):
            complete_kwargs["logical_decision_id"] = logical_decision_id
        try:
            raw = agent.model_client.complete(prompt, agent.max_new_tokens, **complete_kwargs)
        except Exception as exc:
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            provider_metadata = self._provider_metadata(
                agent,
                logical_decision_id,
                status="error",
                error=exc,
                completion_metadata=completion_metadata,
            )
            if completion_metadata:
                prompt_metadata.update(provider_metadata)
                prompt_metadata.update(completion_metadata)
            else:
                prompt_metadata.update(provider_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            attempts = self._provider_attempts(agent, provider_metadata, logical_decision_id)
            self._record_provider_metrics(task_state, provider_metadata, attempts=attempts)
            for attempt in attempts:
                agent.emit_trace(task_state, "provider_attempt", attempt)
            agent.emit_trace(task_state, "provider_response", provider_metadata)
            self._persist_model_failure(task_state, user_message, exc, run_started_at, prompt_metadata)
            raise
        completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
        provider_metadata = self._provider_metadata(
            agent,
            logical_decision_id,
            completion_metadata=completion_metadata,
        )
        if completion_metadata:
            prompt_metadata.update(provider_metadata)
            prompt_metadata.update(completion_metadata)
        else:
            prompt_metadata.update(provider_metadata)
        agent.last_completion_metadata = completion_metadata
        agent.last_prompt_metadata = prompt_metadata
        attempts = self._provider_attempts(agent, provider_metadata, logical_decision_id)
        self._record_provider_metrics(task_state, provider_metadata, attempts=attempts)
        for attempt in attempts:
            agent.emit_trace(task_state, "provider_attempt", attempt)
        agent.emit_trace(task_state, "provider_response", provider_metadata)
        kind, payload = agent.parse(raw, action_chunking=agent.action_chunking)
        agent.emit_trace(
            task_state,
            "model_parsed",
            {
                "kind": kind,
                "completion_metadata": completion_metadata,
                "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                "purpose": purpose,
            },
        )
        discarded = int(agent.resume_state.get("recovery_tail_discarded", 0))
        if discarded:
            agent.emit_trace(
                task_state,
                "recovery_tail_discarded",
                {"count": discarded},
            )
        return raw, kind, payload

    @staticmethod
    def _provider_metadata(agent, logical_decision_id, status="ok", error="", completion_metadata=None):
        metadata = dict(getattr(agent.model_client, "last_provider_metadata", {}) or {})
        completion_metadata = dict(completion_metadata or {})
        metadata.setdefault("request_id", "")
        metadata.setdefault("attempt_id", 1)
        metadata["logical_decision_id"] = logical_decision_id
        metadata.setdefault("provider_requests", 1)
        metadata.setdefault("provider_retries", max(0, int(metadata["provider_requests"]) - 1))
        metadata.setdefault("latency_ms", None)
        if status != "ok" or "status" not in metadata:
            metadata["status"] = status
        if error:
            metadata["error"] = agent.redact_text(str(error))
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
            metadata.setdefault(key, None)
            if metadata[key] is None and completion_metadata.get(key) is not None:
                metadata[key] = completion_metadata[key]
        metadata["usage_missing"] = all(metadata[key] is None for key in ("input_tokens", "output_tokens", "total_tokens"))
        return metadata

    @staticmethod
    def _complete_accepts_logical_decision_id(client):
        try:
            parameters = inspect.signature(client.complete).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == "logical_decision_id" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    @staticmethod
    def _provider_attempts(agent, metadata, logical_decision_id):
        attempts = list(getattr(agent.model_client, "last_provider_attempts", []) or [])
        if not attempts:
            attempts = [dict(metadata)]
        normalized = []
        for attempt in attempts:
            item = dict(attempt)
            item.setdefault("logical_decision_id", logical_decision_id)
            item.setdefault("attempt_id", len(normalized) + 1)
            item.setdefault("provider_requests", len(attempts))
            item.setdefault("provider_retries", max(0, len(attempts) - 1))
            normalized.append(agent.redact_artifact(item))
        return normalized

    @staticmethod
    def _record_provider_metrics(task_state, metadata, attempts=None):
        task_state.record_provider_requests(
            metadata.get("provider_requests", 1),
            metadata.get("provider_retries", 0),
            usage_missing=metadata.get("usage_missing", False),
            attempts=attempts,
        )

    def _finish_success(self, task_state, user_message, final, run_started_at):
        agent = self.agent
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        task_state.finish_success(final)
        agent.promote_durable_memory(user_message, final)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger="run_finished")
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": "run_finished",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        return final

    def run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        agent.memory.set_task_summary(user_message)
        agent.record({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(run_id=agent.new_run_id(), task_id=agent.new_task_id(), user_request=user_message)
        latest_action = latest_committed_action(agent)
        if latest_action:
            task_state.action_seq = int(latest_action.get("action_seq", 0) or 0)
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        agent.current_task_state = task_state
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)
        action_runner = PrimitiveActionRunner(agent, task_state, user_message)
        pending_logical_decision_id = None

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while task_state.tool_steps < agent.max_steps and task_state.attempts < max_attempts:
            task_state.record_attempt()
            if pending_logical_decision_id is None:
                pending_logical_decision_id = f"decision_{task_state.logical_decisions + 1}"
                task_state.record_logical_decision()
            logical_decision_id = pending_logical_decision_id
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            raw, kind, payload = self._request_model(
                task_state,
                user_message,
                prompt,
                prompt_metadata,
                run_started_at,
                purpose="action",
                logical_decision_id=logical_decision_id,
            )

            if kind == "tool":
                result = action_runner.run(
                    Action.from_payload(payload),
                    remaining_budget=agent.max_steps - task_state.tool_steps,
                )
                if not result.result_known:
                    action = result.action
                    if agent.tools.get(action.name, {}).get("risky", True):
                        task_state.stop(
                            STOP_REASON_UNKNOWN_RESULT,
                            final_answer="Stopped because a potentially state-changing tool returned an unknown result.",
                        )
                        break
                    agent.record(
                        {
                            "role": "assistant",
                            "content": (
                                "Runtime observation:\n"
                                f"Action interrupted: {action.name} result is unknown.\n"
                                "Replan using an independent read-only action."
                            ),
                            "created_at": now(),
                        }
                    )
                pending_logical_decision_id = None
                continue

            if kind == "chunk":
                if not agent.action_chunking["enabled"]:
                    agent.record(
                        {
                            "role": "assistant",
                            "content": agent.retry_notice(
                                "action chunks are disabled",
                                allow_chunk=False,
                            ),
                            "created_at": now(),
                        }
                    )
                    agent.run_store.write_task_state(task_state)
                    pending_logical_decision_id = None
                    continue
                validator = ChunkValidator(
                    agent=agent,
                    allowed_tools=agent.action_chunking["allowed_tools"],
                    max_actions_per_chunk=agent.action_chunking["max_actions_per_chunk"],
                )
                execution = ChunkExecutor(
                    runner=action_runner,
                    validator=validator,
                    boundary_policy=BoundaryPolicy(
                        agent.action_chunking["observation_budget_chars"],
                        skill_guidance_enabled=agent.action_chunking["skill_guidance_enabled"],
                    ),
                ).run(
                    payload,
                    remaining_budget=agent.max_steps - task_state.tool_steps,
                )
                task_state.record_chunk(execution.summary)
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(task_state, "chunk_finished", execution.summary.to_dict())
                observation = render_chunk_observation(execution.summary)
                if observation:
                    agent.record(
                        {
                            "role": "assistant",
                            "content": observation,
                            "created_at": now(),
                        }
                    )
                pending_logical_decision_id = None
                continue

            if kind == "retry":
                agent.record({"role": "assistant", "content": payload, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                # 解析失败重试仍属于同一次 logical decision；下一次真实
                # provider 请求会复用同一个 logical_decision_id。
                continue

            final = (payload or raw).strip()
            return self._finish_success(task_state, user_message, final, run_started_at)

        if task_state.tool_steps >= agent.max_steps:
            task_state.record_attempt()
            logical_decision_id = f"decision_{task_state.logical_decisions + 1}"
            task_state.record_logical_decision()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            prompt += (
                "\n\nRuntime notice: the tool budget is exhausted. Do not call another tool. "
                "Use the evidence already present in the tool history and return exactly one "
                "non-empty <final>...</final> answer."
            )
            prompt_metadata["finalization"] = True
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                    "purpose": "finalization",
                },
            )
            raw, kind, payload = self._request_model(
                task_state,
                user_message,
                prompt,
                prompt_metadata,
                run_started_at,
                purpose="finalization",
                logical_decision_id=logical_decision_id,
            )
            if kind == "final":
                final = (payload or raw).strip()
                return self._finish_success(task_state, user_message, final, run_started_at)

        if task_state.stop_reason == STOP_REASON_UNKNOWN_RESULT:
            final = task_state.final_answer
        elif task_state.attempts >= max_attempts and task_state.tool_steps < agent.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        agent.promote_durable_memory(user_message, final)
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        return final
