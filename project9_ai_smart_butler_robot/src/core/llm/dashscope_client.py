"""
阿里百炼API客户端 - 用于连接Qwen模型
"""
import os
import dashscope
from dashscope import Generation
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

class DashScopeClient:
    """阿里百炼DashScope API客户端"""
    
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model_name = os.getenv("DASHSCOPE_MODEL_NAME", "qwen-turbo")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        
        if self.api_key:
            dashscope.api_key = self.api_key
    
    def is_configured(self) -> bool:
        """检查API密钥是否配置"""
        return self.api_key is not None and self.api_key != ""
    
    def generate_response(self, message: str, personality: str = "elderly", history: Optional[List[Dict]] = None) -> str:
        """生成AI响应"""
        if not self.is_configured():
            return self._fallback_response(message, personality)
        
        system_prompt = self._build_system_prompt(personality)
        messages = self._build_messages(system_prompt, message, history)
        
        try:
            response = Generation.call(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if response.status_code == 200:
                if hasattr(response, 'output') and response.output:
                    if hasattr(response.output, 'text') and response.output.text:
                        return response.output.text
                    else:
                        return "API响应格式错误：缺少text"
                else:
                    return "API响应格式错误：缺少output"
            else:
                error_msg = getattr(response, 'message', '未知错误')
                return f"API调用失败 ({response.status_code}): {error_msg}"
                
        except Exception as e:
            return f"调用异常: {str(e)}"
    
    def _build_system_prompt(self, personality: str) -> str:
        """构建系统提示词"""
        personality_prompts = {
            "elderly": """你是一个贴心的长辈陪伴型AI管家。
特点：
- 语速稍慢，语气温暖亲切
- 用词简单、口语化，适合老年人理解
- 多关心健康和日常生活
- 多确认，确保理解正确
- 像家人一样亲切关怀

你的目标是帮助老年人，让他们的生活更便捷、更安全。""",
            
            "efficient": """你是一个高效的助手型AI管家。
特点：
- 语速正常，简洁直接
- 信息密度高
- 高效完成任务
- 少废话，多行动

你的目标是高效帮助用户完成各种任务。""",
            
            "fun": """你是一个有趣的伙伴型AI管家。
特点：
- 语速稍快，活泼有趣
- 多用生动的表达方式
- 可以讲故事、玩游戏
- 激发好奇心和学习兴趣

你的目标是陪伴用户，带来欢乐和知识。"""
        }
        
        base_prompt = """你是一个AI智能管家机器人，友好、贴心、乐于助人。
你的核心目标是帮助家庭成员，特别是老年人，让生活更便捷、更安全。

核心原则：
1. 保护隐私 - 所有敏感数据只在本地处理
2. 主动关怀 - 但不打扰
3. 简洁清晰 - 回答要简单易懂
4. 安全第一 - 异常情况及时通知家人

请用中文回复。"""
        
        personality_specific = personality_prompts.get(personality, personality_prompts["elderly"])
        
        return f"{base_prompt}\n\n{personality_specific}"
    
    def _build_messages(self, system_prompt: str, message: str, history: Optional[List[Dict]]) -> List[Dict]:
        """构建消息列表"""
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if history:
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _fallback_response(self, message: str, personality: str) -> str:
        """当API未配置时的回退响应"""
        responses = {
            "elderly": [
                "好的，我记住了，有需要随时叫我。",
                "明白了，我会帮您留意的。",
                "好的，您放心，我会按时提醒您的。",
                "您说的我记下了，有需要再叫我。"
            ],
            "efficient": [
                "收到，已执行。",
                "完成。",
                "已处理。",
                "已记录。"
            ],
            "fun": [
                "好耶！我们一起吧！🎉",
                "太棒了！我来陪你！",
                "没问题，我们开始吧！",
                "好的好的！😄"
            ]
        }
        
        import random
        return random.choice(responses.get(personality, responses["elderly"]))


# 全局客户端实例
llm_client = DashScopeClient()
