"""Stable prompt prefix construction."""

import hashlib
import json
import textwrap
from dataclasses import dataclass

from .action_chunk import normalize_action_chunking
from .workspace import now


@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


def tool_signature(tools):
    payload = []
    for name in sorted(tools):
        tool = tools[name]
        payload.append(
            {
                "name": name,
                "schema": tool["schema"],
                "risky": tool["risky"],
                "description": tool["description"],
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_prompt_prefix(workspace, tools, built_at=None, action_chunking=None):
    action_chunking = normalize_action_chunking(action_chunking)
    chunk_enabled = action_chunking["enabled"]
    tool_lines = []
    for name, tool in tools.items():
        fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
        risk = "approval required" if tool["risky"] else "safe"
        tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
    tool_text = "\n".join(tool_lines)
    examples = [
        '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
        '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
        '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
        '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    ]
    if chunk_enabled:
        examples.insert(
            2,
            '<chunk>{"actions":[{"name":"read_file","args":{"path":"app/config.py","start":1,"end":80}},{"name":"read_file","args":{"path":"app/parser.py","start":1,"end":80}},{"name":"read_file","args":{"path":"app/loader.py","start":1,"end":80}}]}</chunk>',
        )
    examples.append("<final>Done.</final>")
    examples = "\n".join(examples)
    if chunk_enabled:
        response_rule = "- Return exactly one <tool>...</tool>, one <chunk>...</chunk>, or one <final>...</final>."
        chunk_rules = (
            f"- A chunk may contain at most {action_chunking['max_actions_per_chunk']} ordered read-only actions "
            f"from: {', '.join(action_chunking['allowed_tools'])}."
            " Do not reference another action's result; use only already-known arguments.\n"
            "- When at least two read-only operations have known arguments and are independent of one another, "
            "prefer one <chunk> call.\n"
            "- If any later action argument depends on an earlier observation, do not use <chunk>; issue separate "
            "<tool> calls so the observation can be consumed first."
        )
        if action_chunking["skill_guidance_enabled"]:
            chunk_rules += " Skill guidance is advisory; use boundary_hint=true only at a natural inspection boundary."
    else:
        response_rule = "- Return exactly one <tool>...</tool> or one <final>...</final>."
        chunk_rules = ""
    # prefix 可以理解成 agent 的“工作手册”：
    # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
    text = textwrap.dedent(
        f"""\
        You are pico, a small local coding agent working inside a local repository.

        Rules:
        - Use tools instead of guessing about the workspace.
        {response_rule}
        - Tool calls must look like:
          <tool>{{"name":"tool_name","args":{{...}}}}</tool>
        - For write_file and patch_file with multi-line text, prefer XML style:
          <tool name="write_file" path="file.py"><content>...</content></tool>
        - Final answers must look like:
          <final>your answer</final>
        - Never invent tool results.
        - Keep answers concise and concrete.
        - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
        - Before writing tests for existing code, read the implementation first.
        - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
        - New files should be complete and runnable, including obvious imports.
        - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
        - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.
        {chunk_rules}

        Tools:
        {tool_text}

        Valid response examples:
        {examples}

        {workspace.text()}
        """
    ).strip()
    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
