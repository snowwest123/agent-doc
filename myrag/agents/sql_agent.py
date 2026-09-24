"""Text-to-SQL Agent。

核心能力：
1. 自动感知数据库 schema
2. 多轮工具调用（agent 循环）
3. SQL 自修正（错了让 LLM 重写）
4. 自然语言回答
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from myrag.llm import LLMEndpoint
from myrag.tools.base import ToolRegistry, ToolWrapper


@dataclass
class AgentStep:
    """Agent 执行的每一步记录。"""
    step: int
    action: str       # "tool_call" / "final_answer"
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    reasoning: str = ""
    final_answer: str | None = None


@dataclass
class AgentResult:
    """Agent 最终结果。"""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = True


def _classify_error(error: Exception) -> str:
    """把 Python 异常分类成 LLM 友好的错误信息。

    为什么要分类？
    - LLM 看到具体错误类型才知道下一步该改什么
    - 比如"参数错误"应该改参数名，"SQL 错误"应该改 SQL
    - "权限错误"应该放弃，不该重试
    """
    msg = str(error).lower()

    # 1. LangChain Tool 参数验证错误（最常见）
    if "validation error" in msg or "field required" in msg:
        return (
            f"【参数错误】{error}\n"
            "→ 工具调用参数名/类型不对。请检查工具的 schema 定义，"
            "参数名必须严格匹配（区分大小写）。"
        )

    # 2. SQL 语法错误
    if "syntax error" in msg or "syntax" in msg:
        return (
            f"【SQL 语法错误】{error}\n"
            "→ SQL 写法不对。常见原因：列名错误、关键字拼错、缺少引号。"
            "请重新阅读 schema 后重写 SQL。"
        )

    # 3. SQL 表/列不存在
    if "does not exist" in msg or "not found" in msg or "undefined" in msg:
        return (
            f"【表或列不存在】{error}\n"
            "→ 引用的表名/列名在数据库里找不到。"
            "请用 get_hologres_schema 重新确认表结构。"
        )

    # 4. 权限错误（不可重试）
    if "permission" in msg or "denied" in msg or "forbidden" in msg:
        return (
            f"【权限错误】{error}\n"
            "→ 当前用户没有权限。请检查 RAM 授权或换一个账号。"
            "这类错误重试无意义。"
        )

    # 5. 网络/连接错误
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return (
            f"【连接错误】{error}\n"
            "→ 网络或数据库连接问题。请检查：1) 实例是否运行中 2) 白名单是否包含你的 IP 3) 账号密码是否正确。"
        )

    # 6. 未知错误
    return f"【其他错误】{error}"


class TextToSQLAgent:
    """Text-to-SQL Agent：自然语言 → SQL → 答案。"""

    def __init__(
        self,
        llm: LLMEndpoint,
        tool_registry: ToolRegistry,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    def _build_system_prompt(self, db_path: str, attempted: list | None = None) -> str:
        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}"
            for t in self.tool_registry.get_descriptions()
        )
        attempted_note = ""
        if attempted:
            attempted_note = (
                "\n\n⚠️ **以下尝试已经失败，不要重复**：\n"
                + "\n".join(
                    f"- {a['tool']}({a['args']}) → {a['error']}"
                    for a in attempted[-3:]  # 只显示最近 3 次
                )
                + "\n请基于失败信息调整策略，而不是重复同样的调用。"
            )
        return (
            "你是一个 SQL 数据分析助手，可以调用工具查询数据库。\n\n"
            f"数据库路径：{db_path}\n\n"
            f"可用工具：\n{tools_desc}\n\n"
            "工作流程：\n"
            "1. 如果不确定数据库结构，先用 get_db_schema 看表\n"
            "2. 用 sql_query 查数据\n"
            "3. 如果 SQL 报错，根据错误分类调整：\n"
            "   - 【参数错误】→ 检查工具参数名\n"
            "   - 【SQL 语法错误】→ 重新阅读 schema 后重写\n"
            "   - 【表或列不存在】→ 重新查 schema\n"
            "   - 【权限错误】→ 不要重试，说明用户无权限\n"
            "   - 【连接错误】→ 提示用户检查配置\n"
            "4. 基于查询结果用自然语言回答\n"
            f"{attempted_note}\n\n"
            "输出格式（JSON）：\n"
            "调工具：\n"
            '{"action": "tool_call", "tool": "工具名", "args": {...}, "reasoning": "..."}\n\n'
            "最终答案：\n"
            '{"action": "final_answer", "answer": "...", "reasoning": "..."}\n\n'
            "只输出 JSON，不要其他文字。"
        )

    async def run(self, question: str, db_path: str) -> AgentResult:
        """运行 Agent，并兼容原有的一次性返回接口。"""
        result: AgentResult | None = None
        async for event in self.astream_events(question, db_path):
            if event.get("type") == "done":
                result = AgentResult(
                    answer=event["answer"],
                    steps=[AgentStep(**step) for step in event["steps"]],
                    success=event["success"],
                )
        return result or AgentResult(answer="Agent 未能完成任务", success=False)

    async def astream_events(self, question: str, db_path: str):
        """流式返回 Agent 的工具调用、工具结果和最终答案事件。"""
        system_prompt = self._build_system_prompt(db_path)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        steps: list[AgentStep] = []
        final_answer = None
        attempted: list[dict] = []
        yield {"type": "agent_start", "question": question}

        for i in range(self.max_iterations):
            yield {"type": "llm_start", "step": i + 1, "message": "正在分析下一步 Agent 动作"}
            response_text = (await self.llm.ainvoke(messages)).strip()
            decision = self._parse_decision(response_text)
            yield {"type": "llm_decision", "step": i + 1, "action": decision.get("action") if decision else "final_answer"}
            if not decision:
                final_answer = response_text
                step = AgentStep(step=i + 1, action="final_answer", final_answer=response_text)
                steps.append(step)
                yield {"type": "final_answer", "answer": final_answer, "step": i + 1}
                break

            action = decision.get("action")
            if action == "final_answer":
                final_answer = decision.get("answer", "")
                step = AgentStep(
                    step=i + 1,
                    action="final_answer",
                    final_answer=final_answer,
                    reasoning=decision.get("reasoning", ""),
                )
                steps.append(step)
                yield {"type": "final_answer", "answer": final_answer, "step": i + 1}
                break

            if action != "tool_call":
                continue

            tool_name = decision.get("tool")
            tool_args = decision.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            wrapper = self.tool_registry.get(tool_name)
            args_schema = getattr(getattr(wrapper, "tool", None), "args_schema", None)
            fields = getattr(args_schema, "model_fields", None) if args_schema else None
            if fields is None:
                fields = getattr(args_schema, "__fields__", {}) if args_schema else {}
            if "db_path" in fields and "db_path" not in tool_args:
                tool_args["db_path"] = db_path

            yield {
                "type": "tool_call",
                "step": i + 1,
                "tool": tool_name,
                "args": tool_args,
                "reasoning": decision.get("reasoning", ""),
            }
            if not wrapper:
                tool_result = f"【未知工具】{tool_name} 不存在。可用工具：{self.tool_registry.list_tools()}"
                attempted.append({"tool": tool_name, "args": tool_args, "error": tool_result})
            else:
                try:
                    raw_result = await wrapper.tool.ainvoke(tool_args)
                    tool_result = wrapper.format_output(raw_result)
                except Exception as e:
                    tool_result = _classify_error(e)
                    attempted.append({"tool": tool_name, "args": tool_args, "error": tool_result[:200]})

            step = AgentStep(
                step=i + 1,
                action="tool_call",
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                reasoning=decision.get("reasoning", ""),
            )
            steps.append(step)
            yield {"type": "tool_result", "step": i + 1, "tool": tool_name, "result": tool_result}
            messages.append(SystemMessage(content=f"工具 {tool_name} 返回：\n{tool_result}"))
            messages[0] = SystemMessage(content=self._build_system_prompt(db_path, attempted))

        if not final_answer:
            last_step = steps[-1] if steps else None
            if last_step and last_step.tool_result and "【错误】" not in last_step.tool_result and "【" not in last_step.tool_result:
                messages.append(HumanMessage(content="请基于工具结果直接给出最终答案，不要再调用工具。"))
                summary = (await self.llm.ainvoke(messages)).strip()
                decision = self._parse_decision(summary)
                final_answer = decision.get("answer", "") if decision and decision.get("action") == "final_answer" else summary
            else:
                final_answer = "Agent 未能完成任务（可能因为重复的工具错误）"
            yield {"type": "final_answer", "answer": final_answer, "step": len(steps) + 1}

        yield {
            "type": "done",
            "answer": final_answer,
            "success": bool(final_answer) and "未能" not in final_answer,
            "steps": [step.__dict__ for step in steps],
        }

    @staticmethod
    def _parse_decision(text: str) -> dict | None:
        """从 LLM 输出里抽 JSON 决策。"""
        # 找 ```json ... ``` 块
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # 找 { ... } 块
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return None
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None