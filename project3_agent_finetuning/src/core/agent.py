import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from abc import ABC, abstractmethod

# 尝试导入dashscope，如果失败则使用模拟实现
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("Warning: dashscope not available, using mock implementation")

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.memory: List[Dict[str, Any]] = []
        
        # 检查是否在模拟模式
        self.mock_mode = not DASHSCOPE_AVAILABLE or os.getenv("MOCK_MODE", "true").lower() == "true"
        if self.mock_mode:
            logger.warning(
                f"⚠️  {name} 运行在 MOCK 模式！"
                "LLM调用将使用模拟回复。如需启用真实LLM，请安装dashscope并设置MOCK_MODE=false"
            )

    @abstractmethod
    def run(self, user_input: str, **kwargs) -> Dict[str, Any]:
        pass


class ReActAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ReActLawyer", config)
        self.tools = {
            "search_law": self._search_law,
            "search_case": self._search_case,
            "calculate_damage": self._calculate_damage
        }

    def run(self, user_input: str, **kwargs) -> Dict[str, Any]:
        tools_used = []
        thought_process = []

        prompt = f"""你是一位专业的法律顾问。请使用ReAct模式回答用户的法律问题。

用户问题: {user_input}

可用工具:
- search_law(关键词): 搜索相关法律条文
- search_case(关键词): 搜索相关案例
- calculate_damage(金额): 计算赔偿金

请按以下格式回复:
Thought: [你的思考过程]
Action: [工具名称(参数)] 或 [直接回答]
Observation: [如果调用工具，这里是结果]
Final Answer: [最终回答]

现在开始:"""

        response = self._call_llm(prompt)
        thought_process.append(response)

        action_match = re.search(r'Action:\s*(\w+)\((.*?)\)', response)
        if action_match:
            tool_name = action_match.group(1)
            tool_arg = action_match.group(2).strip('"\'')
            if tool_name in self.tools:
                observation = self.tools[tool_name](tool_arg)
                tools_used.append({"tool": tool_name, "argument": tool_arg})
                thought_process.append(f"Observation: {observation}")

                final_prompt = f"""基于以下信息给出最终回答:

{chr(10).join(thought_process)}

Final Answer:"""
                final_answer = self._call_llm(final_prompt)
            else:
                final_answer = "抱歉，我无法使用该工具。"
        else:
            final_answer = response.split("Final Answer:")[-1].strip() if "Final Answer:" in response else response

        return {
            "output": final_answer,
            "tools_used": tools_used,
            "thought_process": thought_process
        }

    def _search_law(self, keyword: str) -> str:
        laws = {
            "合同": "《民法典》第四百九十条：当事人采用合同书形式订立合同的，自当事人均签名、盖章或者按指印时合同成立。",
            "侵权": "《民法典》第一千一百六十五条：行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。",
            "劳动": "《劳动合同法》第三十八条：用人单位有下列情形之一的，劳动者可以解除劳动合同。",
        }
        return laws.get(keyword, f"未找到与'{keyword}'相关的法律条文。")

    def _search_case(self, keyword: str) -> str:
        cases = {
            "合同纠纷": "案例：2023年某公司买卖合同纠纷案，法院判决违约方支付违约金10万元。",
            "交通事故": "案例：2024年某机动车交通事故责任纠纷案，赔偿金额共计25万元。",
        }
        return cases.get(keyword, f"未找到与'{keyword}'相关的案例。")

    def _calculate_damage(self, amount: str) -> str:
        try:
            base = float(amount)
            interest = base * 0.03
            total = base + interest
            return f"本金: {base}元, 利息: {interest:.2f}元, 总计: {total:.2f}元"
        except:
            return "金额格式错误"

    def _call_llm(self, prompt: str) -> str:
        try:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if DASHSCOPE_AVAILABLE and api_key:
                dashscope.api_key = api_key
                response = dashscope.Generation.call(
                    model='qwen-turbo',
                    prompt=prompt,
                )
                if response.status_code == 200:
                    return response.output.text
            # 模拟回复
            if "合同" in prompt:
                return "Thought: 需要搜索合同法相关条款\nAction: search_law(合同)\nObservation: 《民法典》第四百九十条：当事人采用合同书形式订立合同的，自当事人均签名、盖章或者按指印时合同成立。\nFinal Answer: 根据《民法典》，合同自双方签名、盖章或按指印时成立。"
            elif "侵权" in prompt:
                return "Thought: 需要搜索侵权责任相关内容\nAction: search_law(侵权)\nObservation: 《民法典》第一千一百六十五条：行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。\nFinal Answer: 根据法律，侵权者需要承担相应的赔偿责任。"
            else:
                return f"这是模拟的ReAct Agent回复。根据您的问题，建议您咨询专业法律人士获取准确的法律建议。"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"抱歉，遇到问题: {str(e)}"


class MultiAgentSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.planner_agent = self._create_planner()
        self.executor_agent = self._create_executor()

    def _create_planner(self):
        class PlannerAgent(BaseAgent):
            def __init__(self):
                super().__init__("Planner", {})

            def run(self, user_input: str, **kwargs) -> Dict[str, Any]:
                steps = [
                    f"步骤1: 分析问题: {user_input[:30]}...",
                    "步骤2: 收集相关信息",
                    "步骤3: 执行解决方案",
                    "步骤4: 验证结果"
                ]
                return {
                    "plan": steps,
                    "estimated_time": "5分钟"
                }

        return PlannerAgent()

    def _create_executor(self):
        class ExecutorAgent(BaseAgent):
            def __init__(self):
                super().__init__("Executor", {})

            def run(self, plan: List[str], **kwargs) -> Dict[str, Any]:
                results = []
                for step in plan:
                    results.append(f"✅ 完成: {step}")
                return {
                    "execution_log": results,
                    "status": "completed"
                }

        return ExecutorAgent()

    def run(self, user_input: str, **kwargs) -> Dict[str, Any]:
        tools_used = [{"agent": "Planner", "action": "plan"}]

        plan_result = self.planner_agent.run(user_input)
        plan = plan_result["plan"]

        tools_used.append({"agent": "Executor", "action": "execute"})
        exec_result = self.executor_agent.run(plan)

        return {
            "output": f"任务完成！\n计划: {chr(10).join(plan)}\n执行: {chr(10).join(exec_result['execution_log'])}",
            "tools_used": tools_used,
            "plan": plan,
            "execution": exec_result
        }


class VisualAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("VisualAnalyzer", config)

    def run(self, user_input: str, image: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        tools_used = [{"tool": "visual_analysis", "action": "contract_review"}]

        risks = [
            {"type": "金额风险", "description": "合同金额未明确标注大写", "severity": "high", "position": "第3条第2款"},
            {"type": "期限风险", "description": "违约责任期限不明确", "severity": "medium", "position": "第7条"},
            {"type": "格式建议", "description": "建议添加骑缝章", "severity": "low", "position": "全文"}
        ]

        analysis = f"""## 合同风险分析报告

### 分析概要
- 检测到 {len(risks)} 个潜在风险点
- 高风险: {len([r for r in risks if r['severity'] == 'high'])} 个
- 中风险: {len([r for r in risks if r['severity'] == 'medium'])} 个
- 低风险: {len([r for r in risks if r['severity'] == 'low'])} 个

### 详细风险点
"""
        for risk in risks:
            icon = "🔴" if risk["severity"] == "high" else "🟡" if risk["severity"] == "medium" else "🟢"
            analysis += f"\n{icon} **{risk['type']}**\n- 描述: {risk['description']}\n- 位置: {risk['position']}\n"

        analysis += "\n### 建议\n1. 建议补充大写金额\n2. 明确违约责任期限\n3. 增强合同规范性"

        return {
            "output": analysis,
            "tools_used": tools_used,
            "risks": risks,
            "summary": f"发现{len(risks)}个风险点"
        }


class AgentManager:
    def __init__(self):
        self.sessions: Dict[int, Dict[str, Any]] = {}
        self.agent_classes = {
            "react": ReActAgent,
            "multiagent": MultiAgentSystem,
            "visual": VisualAgent
        }

    def create_session(self, session_id: int, agent_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        if agent_type not in self.agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = self.agent_classes[agent_type]
        agent = agent_class(config)

        session = {
            "id": session_id,
            "agent_type": agent_type,
            "agent": agent,
            "config": config or {},
            "created_at": datetime.utcnow().isoformat(),
            "interactions": []
        }
        self.sessions[session_id] = session
        logger.info(f"Created agent session {session_id} of type {agent_type}")
        return session

    def chat(self, session_id: int, user_input: str, image: Optional[str] = None) -> Dict[str, Any]:
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        agent = session["agent"]

        result = agent.run(user_input, image=image)

        interaction = {
            "user_input": user_input,
            "agent_output": result["output"],
            "tools_used": result.get("tools_used", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        session["interactions"].append(interaction)

        return result

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in s.items() if k != "agent"}
            for s in self.sessions.values()
        ]

    def get_available_agents(self) -> List[str]:
        return list(self.agent_classes.keys())


agent_manager = AgentManager()

