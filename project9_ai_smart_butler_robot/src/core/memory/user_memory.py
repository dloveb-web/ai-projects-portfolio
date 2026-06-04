"""
用户习惯记忆和摘要模块
用于记录和分析用户的日常活动、偏好、习惯等
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict


@dataclass
class InteractionRecord:
    """交互记录"""
    timestamp: str
    interaction_type: str  # 对话、控制设备、提醒等
    content: str
    user_id: Optional[int] = None
    emotional_state: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DailyRoutine:
    """日常作息记录"""
    wake_up_time: Optional[str] = None  # 起床时间
    sleep_time: Optional[str] = None  # 睡觉时间
    meal_times: List[str] = None  # 用餐时间
    exercise_times: List[str] = None  # 锻炼时间
    medicine_times: List[str] = None  # 用药时间
    
    def __post_init__(self):
        if self.meal_times is None:
            self.meal_times = []
        if self.exercise_times is None:
            self.exercise_times = []
        if self.medicine_times is None:
            self.medicine_times = []


@dataclass
class Preference:
    """用户偏好"""
    personality_mode: str = "elderly"  # 偏好的人格模式
    language_style: str = "polite"  # 语言风格
    preferred_topics: List[str] = None  # 喜欢的话题
    disliked_topics: List[str] = None  # 不喜欢的话题
    volume_level: str = "medium"  # 音量偏好
    speech_speed: str = "normal"  # 语速偏好
    
    def __post_init__(self):
        if self.preferred_topics is None:
            self.preferred_topics = []
        if self.disliked_topics is None:
            self.disliked_topics = []


class UserMemory:
    """用户记忆管理器"""
    
    def __init__(self, data_dir: str = "data/memory"):
        self.data_dir = data_dir
        self.interactions: List[InteractionRecord] = []
        self.daily_routine = DailyRoutine()
        self.preferences = Preference()
        self.user_profiles: Dict[int, Dict] = {}  # 用户ID -> 用户档案
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载已有数据
        self._load_data()
    
    def record_interaction(self, interaction_type: str, content: str, 
                          user_id: Optional[int] = None,
                          emotional_state: Optional[str] = None) -> None:
        """记录一次交互"""
        record = InteractionRecord(
            timestamp=datetime.now().isoformat(),
            interaction_type=interaction_type,
            content=content,
            user_id=user_id,
            emotional_state=emotional_state
        )
        self.interactions.append(record)
        self._save_data()
        
        # 更新日常作息
        self._update_daily_routine(interaction_type, content)
    
    def _update_daily_routine(self, interaction_type: str, content: str) -> None:
        """根据交互更新日常作息"""
        current_time = datetime.now().strftime("%H:%M")
        
        if interaction_type == "wake_up":
            self.daily_routine.wake_up_time = current_time
        elif interaction_type == "sleep":
            self.daily_routine.sleep_time = current_time
        elif interaction_type == "meal":
            self.daily_routine.meal_times.append(current_time)
        elif interaction_type == "medicine":
            self.daily_routine.medicine_times.append(current_time)
        elif interaction_type == "exercise":
            self.daily_routine.exercise_times.append(current_time)
    
    def update_preference(self, key: str, value: Any) -> None:
        """更新用户偏好"""
        if hasattr(self.preferences, key):
            setattr(self.preferences, key, value)
            self._save_data()
    
    def add_user_profile(self, user_id: int, profile: Dict) -> None:
        """添加用户档案"""
        self.user_profiles[user_id] = {
            **profile,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        self._save_data()
    
    def generate_summary(self, user_id: Optional[int] = None) -> Dict:
        """生成用户习惯摘要"""
        now = datetime.now()
        
        # 过滤指定用户的交互（如果有用户ID）
        user_interactions = self.interactions
        if user_id:
            user_interactions = [i for i in self.interactions if i.user_id == user_id]
        
        # 统计信息
        interaction_counts = defaultdict(int)
        for i in user_interactions:
            interaction_counts[i.interaction_type] += 1
        
        # 分析活跃时间段
        active_hours = self._analyze_active_hours(user_interactions)
        
        # 情感分析统计
        emotion_counts = Counter([i.emotional_state for i in user_interactions if i.emotional_state])
        
        # 生成作息建议
        routine_suggestions = self._generate_routine_suggestions()
        
        summary = {
            "generated_at": now.isoformat(),
            "time_period": self._get_time_period(user_interactions),
            "total_interactions": len(user_interactions),
            "interaction_breakdown": dict(interaction_counts),
            "active_hours": active_hours,
            "emotional_pattern": dict(emotion_counts),
            "daily_routine": asdict(self.daily_routine),
            "preferences": asdict(self.preferences),
            "routine_suggestions": routine_suggestions,
            "summary_text": self._generate_natural_summary()
        }
        
        return summary
    
    def _analyze_active_hours(self, interactions: List[InteractionRecord]) -> Dict:
        """分析活跃时间段"""
        hour_counts = defaultdict(int)
        
        for i in interactions:
            try:
                dt = datetime.fromisoformat(i.timestamp)
                hour_counts[dt.hour] += 1
            except:
                pass
        
        if hour_counts:
            max_hour = max(hour_counts, key=hour_counts.get)
            return {
                "most_active_hour": f"{max_hour:02d}:00",
                "hourly_distribution": dict(hour_counts)
            }
        return {}
    
    def _generate_routine_suggestions(self) -> List[str]:
        """生成作息建议"""
        suggestions = []
        
        # 基于当前数据生成建议
        routine = self.daily_routine
        
        if routine.wake_up_time:
            suggestions.append(f"起床时间相对稳定，建议保持 {routine.wake_up_time} 左右起床")
        
        if routine.medicine_times:
            # 分析用药时间规律
            if len(routine.medicine_times) >= 2:
                suggestions.append("记得按时服药，已记录用药规律")
        
        if routine.sleep_time:
            suggestions.append(f"保持规律睡眠，建议 {routine.sleep_time} 左右休息")
        
        # 默认建议
        if not suggestions:
            suggestions = [
                "建议保持规律作息",
                "按时用药很重要",
                "适当进行轻量活动"
            ]
        
        return suggestions
    
    def _generate_natural_summary(self) -> str:
        """生成自然语言摘要"""
        routine = self.daily_routine
        parts = []
        
        parts.append("根据记录，您的日常习惯如下：")
        parts.append("")
        
        if routine.wake_up_time:
            parts.append(f"• 通常在 {routine.wake_up_time} 左右起床")
        
        if routine.sleep_time:
            parts.append(f"• 通常在 {routine.sleep_time} 左右休息")
        
        if routine.meal_times:
            parts.append(f"• 用餐时间：{', '.join(routine.meal_times[-3:])}")
        
        if routine.medicine_times:
            parts.append(f"• 用药时间：{', '.join(routine.medicine_times[-3:])}")
        
        if self.preferences.preferred_topics:
            parts.append(f"• 喜欢的话题：{', '.join(self.preferences.preferred_topics[:3])}")
        
        parts.append("")
        parts.append("我会根据您的习惯提供更好的服务！")
        
        return "\n".join(parts)
    
    def _get_time_period(self, interactions: List[InteractionRecord]) -> str:
        """获取记录覆盖的时间段"""
        if not interactions:
            return "无记录"
        
        try:
            times = [datetime.fromisoformat(i.timestamp) for i in interactions]
            start = min(times).strftime("%Y-%m-%d")
            end = max(times).strftime("%Y-%m-%d")
            return f"{start} 至 {end}"
        except:
            return "时间段记录不完整"
    
    def _load_data(self) -> None:
        """从文件加载数据"""
        memory_file = os.path.join(self.data_dir, "user_memory.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 加载交互记录
                    self.interactions = [
                        InteractionRecord(**r) for r in data.get("interactions", [])
                    ]
                    
                    # 加载日常作息
                    if "daily_routine" in data:
                        self.daily_routine = DailyRoutine(**data["daily_routine"])
                    
                    # 加载偏好
                    if "preferences" in data:
                        self.preferences = Preference(**data["preferences"])
                    
                    # 加载用户档案
                    self.user_profiles = data.get("user_profiles", {})
            except Exception as e:
                print(f"加载记忆数据失败: {e}")
    
    def _save_data(self) -> None:
        """保存数据到文件"""
        memory_file = os.path.join(self.data_dir, "user_memory.json")
        
        # 手动转换记录为字典
        interactions_data = []
        for r in self.interactions[-1000:]:
            interactions_data.append({
                "timestamp": r.timestamp,
                "interaction_type": r.interaction_type,
                "content": r.content,
                "user_id": r.user_id,
                "emotional_state": r.emotional_state,
                "confidence": r.confidence
            })
        
        routine_data = {
            "wake_up_time": self.daily_routine.wake_up_time,
            "sleep_time": self.daily_routine.sleep_time,
            "meal_times": self.daily_routine.meal_times,
            "exercise_times": self.daily_routine.exercise_times,
            "medicine_times": self.daily_routine.medicine_times
        }
        
        preferences_data = {
            "personality_mode": self.preferences.personality_mode,
            "language_style": self.preferences.language_style,
            "preferred_topics": self.preferences.preferred_topics,
            "disliked_topics": self.preferences.disliked_topics,
            "volume_level": self.preferences.volume_level,
            "speech_speed": self.preferences.speech_speed
        }
        
        data = {
            "interactions": interactions_data,
            "daily_routine": routine_data,
            "preferences": preferences_data,
            "user_profiles": self.user_profiles,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆数据失败: {e}")


# 全局记忆管理器实例
user_memory = UserMemory()
