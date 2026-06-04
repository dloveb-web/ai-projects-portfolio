#!/usr/bin/env python3
"""
用户习惯记忆功能演示
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.memory.user_memory import UserMemory, DailyRoutine, Preference
from datetime import datetime, timedelta
import random

def main():
    print('=' * 60)
    print('🧠 AI智能管家 - 用户习惯记忆功能演示')
    print('=' * 60)
    print()
    
    print('1️⃣ 初始化用户记忆系统...')
    memory = UserMemory()
    
    # 清除旧数据
    memory.interactions = []
    memory.daily_routine = DailyRoutine()
    memory.preferences = Preference()
    
    print('✅ 记忆系统已初始化')
    print()
    
    # 设置基础习惯
    print('2️⃣ 设置用户日常习惯...')
    memory.daily_routine.wake_up_time = '07:30'
    memory.daily_routine.sleep_time = '21:00'
    memory.daily_routine.meal_times = ['08:00', '12:30', '18:30']
    memory.daily_routine.medicine_times = ['08:00', '18:00']
    memory.daily_routine.exercise_times = ['09:30']
    memory.preferences.preferred_topics = ['健康', '天气', '家庭']
    print('✅ 习惯数据已设置')
    print()
    
    # 添加模拟交互记录
    print('3️⃣ 添加模拟交互记录...')
    for i in range(50):
        hour = random.randint(7, 21)
        ts = datetime.now() - timedelta(days=random.randint(0, 7), hours=hour)
        types = ['chat', 'chat', 'chat', 'control', 'reminder']
        itype = random.choice(types)
        
        # 创建记录
        class Record:
            def __init__(self, ts, itype, content, uid):
                self.timestamp = ts
                self.interaction_type = itype
                self.content = content
                self.user_id = uid
                self.emotional_state = None
                self.confidence = 0.0
        
        memory.interactions.append(Record(
            ts=ts.isoformat(),
            itype=itype,
            content=f'模拟交互 {i+1}',
            uid=1
        ))
    
    print(f'✅ 已添加 {len(memory.interactions)} 条交互记录')
    print()
    
    # 保存数据
    print('4️⃣ 保存数据到本地...')
    memory._save_data()
    print('✅ 数据已保存')
    print()
    
    # 测试生成摘要
    print('=' * 60)
    print('📊 生成用户习惯记忆摘要')
    print('=' * 60)
    
    summary = memory.generate_summary(user_id=1)
    
    print()
    print(f'📅 时间段: {summary.get('time_period')}')
    print(f'💬 总交互: {summary.get('total_interactions')} 次')
    
    print()
    print('📋 交互类型分布:')
    for itype, count in summary.get('interaction_breakdown', {}).items():
        print(f'  • {itype}: {count}')
    
    print()
    print('⏰ 活跃时段分析:')
    active_hours = summary.get('active_hours', {})
    if active_hours:
        print(f'  最活跃: {active_hours.get('most_active_hour')}')
    
    print()
    print('🏠 日常作息习惯:')
    routine = summary.get('daily_routine', {})
    if routine.get('wake_up_time'):
        print(f'  • 起床: {routine.get('wake_up_time')}')
    if routine.get('sleep_time'):
        print(f'  • 休息: {routine.get('sleep_time')}')
    if routine.get('medicine_times'):
        print(f'  • 用药: {', '.join(routine.get('medicine_times', [])[:3])}')
    
    print()
    print('💡 作息建议:')
    for suggestion in summary.get('routine_suggestions', [])[:3]:
        print(f'  • {suggestion}')
    
    print()
    print('=' * 60)
    print('📖 自然语言记忆摘要:')
    print('=' * 60)
    print(summary.get('summary_text'))
    print()
    print('=' * 60)
    print('✅ 演示完成！')
    print('📁 记忆数据已保存到: data/memory/user_memory.json')
    print()
    print('现在可以去界面上点击"📊 生成习惯摘要"按钮查看效果！')

if __name__ == '__main__':
    main()
