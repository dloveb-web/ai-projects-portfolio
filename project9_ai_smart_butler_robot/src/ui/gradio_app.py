"""
AI智能管家机器人 - 端侧前端界面模拟
"""
import gradio as gr
import random
import sys
import os
import time
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入LLM客户端
from core.llm.dashscope_client import llm_client

# 导入摄像头控制模块
try:
    from core.vision.camera import camera_controller, image_analyzer, simulate_robot_movement
    CAMERA_AVAILABLE = True
except:
    CAMERA_AVAILABLE = False

# 导入用户记忆模块
try:
    from core.memory.user_memory import user_memory
    MEMORY_AVAILABLE = True
except:
    MEMORY_AVAILABLE = False

SIMULATED_USERS = [
    {"id": 1, "name": "张奶奶", "role": "elderly", "age": 78},
    {"id": 2, "name": "王先生", "role": "adult", "age": 45},
    {"id": 3, "name": "小明", "role": "child", "age": 7}
]

SIMULATED_REMINDERS = [
    {"id": 1, "title": "吃药提醒", "user_id": 1, "medication": "降压药", "time": "08:00", "status": "pending"},
    {"id": 2, "title": "吃药提醒", "user_id": 1, "medication": "降糖药", "time": "12:00", "status": "pending"},
    {"id": 3, "title": "复诊提醒", "user_id": 1, "time": "明天 09:00", "status": "pending"},
    {"id": 4, "title": "会议提醒", "user_id": 2, "time": "14:00", "status": "pending"},
    {"id": 5, "title": "讲故事", "user_id": 3, "time": "19:00", "status": "pending"}
]

SIMULATED_DEVICES = [
    {"id": 1, "name": "客厅灯", "type": "light", "status": "off"},
    {"id": 2, "name": "空调", "type": "ac", "status": "off", "temperature": 26},
    {"id": 3, "name": "电视", "type": "tv", "status": "off"},
    {"id": 4, "name": "卧室灯", "type": "light", "status": "on"},
    {"id": 5, "name": "空气净化器", "type": "purifier", "status": "on"}
]

SIMULATED_HEALTH_DATA = {
    "temperature": 36.5,
    "heart_rate": 72,
    "breathing_rate": 16,
    "oxygen_level": 98
}

PERSONALITY_TYPES = {
    "elderly": {"name": "长辈陪伴型", "description": "语气温和、语速稍慢、字体较大", "icon": "👵"},
    "efficient": {"name": "高效助手型", "description": "简洁直接、信息密度高", "icon": "💼"},
    "fun": {"name": "趣味伙伴型", "description": "活泼有趣、适合儿童", "icon": "🎈"}
}

def generate_response(message: str, personality: str) -> str:
    message_lower = message.lower()
    
    if any(keyword in message_lower for keyword in ["你是谁", "谁", "身份"]):
        responses = {
            "elderly": ["我是您的AI智能管家，专门陪伴您，有什么事随时叫我。", 
                       "您好，我是您的智能管家，很高兴为您服务。"],
            "efficient": ["我是AI智能管家，提供语音交互、智能家居控制、健康监测等服务。"],
            "fun": ["我是你的智能小伙伴！可以陪你聊天、讲故事、玩游戏！🎉"]
        }
        return random.choice(responses.get(personality, responses["elderly"]))
    
    if any(keyword in message_lower for keyword in ["做什么", "可以帮", "功能", "能力"]):
        responses = {
            "elderly": ["我可以帮您提醒吃药、控制家电、监测健康，还能陪您聊天呢。",
                       "我能做很多事：提醒您按时吃药、开关电器、测量健康数据，您需要什么帮助？"],
            "efficient": ["我可以进行语音交互、智能家居控制、健康监测、日程提醒等。"],
            "fun": ["我可以讲故事、玩游戏、回答问题，还能帮你控制家里的智能设备！"]
        }
        return random.choice(responses.get(personality, responses["elderly"]))
    
    if any(keyword in message_lower for keyword in ["你好", "您好", "嗨", "哈喽"]):
        responses = {
            "elderly": ["你好呀！有什么需要帮助的吗？", "您好！很高兴见到您。"],
            "efficient": ["您好，请问有什么需要？"],
            "fun": ["嗨！很高兴认识你！😄"]
        }
        return random.choice(responses.get(personality, responses["elderly"]))
    
    if any(keyword in message_lower for keyword in ["谢谢", "感谢"]):
        responses = {
            "elderly": ["不用客气，这是我应该做的。", "您太客气了，能帮到您我很开心。"],
            "efficient": ["不客气。"],
            "fun": ["不客气！随时找我玩！🎉"]
        }
        return random.choice(responses.get(personality, responses["elderly"]))
    
    if any(keyword in message_lower for keyword in ["天气", "气温"]):
        responses = {
            "elderly": ["今天天气不错，适合出去走走。", "天气有点凉，记得多加件衣服。"],
            "efficient": ["当前气温约26°C，天气晴朗。"],
            "fun": ["今天天气超好！☀️ 我们出去玩吧！"]
        }
        return random.choice(responses.get(personality, responses["elderly"]))
    
    if any(keyword in message_lower for keyword in ["讲故事", "故事"]):
        responses = {
            "elderly": ["好的，我来给您讲一个有趣的故事...", "想听什么类型的故事呢？"],
            "efficient": ["已准备讲故事功能。"],
            "fun": ["好耶！我来讲一个超级有趣的故事！🎈"]
        }
        return random.choice(responses.get(personality, responses["elderly"]))
    
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
    return random.choice(responses.get(personality, responses["elderly"]))

def chat_with_robot(message: str, history: List[Dict], personality: str) -> tuple:
    # 记录用户交互到记忆系统
    if MEMORY_AVAILABLE:
        user_memory.record_interaction(
            interaction_type="chat",
            content=message,
            user_id=1  # 默认用户
        )
    
    if llm_client.is_configured():
        bot_response = llm_client.generate_response(message, personality, history)
    else:
        bot_response = generate_response(message, personality)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_response})
    return "", history

def switch_personality(personality: str):
    info = PERSONALITY_TYPES.get(personality, PERSONALITY_TYPES["elderly"])
    return gr.update(label=f"{info['icon']} {info['name']}"), info["description"]

def toggle_device(device_id: int, devices: List[Dict]) -> list:
    for device in devices:
        if device["id"] == device_id:
            device["status"] = "on" if device["status"] == "off" else "off"
    return devices

def confirm_reminder(reminder_id: int, reminders: List[Dict]) -> list:
    for reminder in reminders:
        if reminder["id"] == reminder_id:
            reminder["status"] = "confirmed"
    return reminders

def get_current_time() -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

def update_health_data() -> dict:
    return {
        "temperature": round(36.0 + random.uniform(0, 1.0), 1),
        "heart_rate": random.randint(60, 85),
        "breathing_rate": random.randint(12, 20),
        "oxygen_level": random.randint(95, 100)
    }

def trigger_emergency() -> str:
    return f"🚨 紧急求助已触发！\n时间: {get_current_time()}\n正在通知紧急联系人..."

def create_robot_ui():
    with gr.Blocks(title="AI智能管家机器人") as demo:
        gr.Markdown("# 🤖 AI智能管家机器人")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 💬 语音交互")
                chatbot = gr.Chatbot(height=400)
                msg_input = gr.Textbox(placeholder="说点什么吧...")
                send_btn = gr.Button("发送", variant="primary")
                
                send_btn.click(
                    chat_with_robot,
                    inputs=[msg_input, chatbot, gr.State("elderly")],
                    outputs=[msg_input, chatbot]
                )
                msg_input.submit(
                    chat_with_robot,
                    inputs=[msg_input, chatbot, gr.State("elderly")],
                    outputs=[msg_input, chatbot]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 功能面板")
                
                gr.Markdown("### 人格模式")
                personality_dropdown = gr.Dropdown(
                    choices=[
                        ("👵 长辈陪伴型", "elderly"),
                        ("💼 高效助手型", "efficient"),
                        ("🎈 趣味伙伴型", "fun")
                    ],
                    value="elderly",
                    label="当前人格"
                )
                personality_desc = gr.Textbox(
                    label="人格描述",
                    value=PERSONALITY_TYPES["elderly"]["description"],
                    interactive=False
                )
                
                personality_dropdown.change(
                    switch_personality,
                    inputs=[personality_dropdown],
                    outputs=[personality_dropdown, personality_desc]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 紧急求助")
                emergency_btn = gr.Button("🆘 紧急呼叫", variant="stop")
                emergency_output = gr.Textbox(label="紧急状态", interactive=False)
                emergency_btn.click(trigger_emergency, outputs=[emergency_output])
                
                gr.Markdown("---")
                gr.Markdown("### 🏠 智能看护")
                location_dropdown = gr.Dropdown(
                    choices=[
                        "🛏️ 卧室",
                        "🛋️ 客厅",
                        "🍳 厨房",
                        "🚿 卫生间"
                    ],
                    value="🛏️ 卧室",
                    label="选择检查位置"
                )
                execute_check_btn = gr.Button("🔍 执行检查", variant="primary")
                
                care_log = gr.Textbox(
                    label="行动日志",
                    value="等待执行检查...",
                    lines=8,
                    interactive=False
                )
                
                analysis_result = gr.Textbox(
                    label="分析结果",
                    value="",
                    lines=4,
                    interactive=False
                )
                
                execute_check_btn.click(
                    execute_care_check,
                    inputs=[location_dropdown],
                    outputs=[care_log, analysis_result]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🧠 用户习惯记忆")
                
                generate_summary_btn = gr.Button("📊 生成习惯摘要", variant="secondary")
                
                summary_display = gr.Textbox(
                    label="习惯摘要",
                    value="点击按钮生成您的日常习惯摘要...",
                    lines=12,
                    interactive=False
                )
                
                summary_details = gr.Textbox(
                    label="详细分析",
                    value="",
                    lines=8,
                    interactive=False
                )
                
                generate_summary_btn.click(
                    generate_memory_summary,
                    outputs=[summary_display, summary_details]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📝 记录活动")
                
                activity_type_dropdown = gr.Dropdown(
                    choices=[
                        "起床",
                        "用餐",
                        "用药",
                        "锻炼",
                        "休息"
                    ],
                    value="起床",
                    label="活动类型"
                )
                
                record_activity_btn = gr.Button("✅ 记录活动")
                
                activity_log = gr.Textbox(
                    label="记录状态",
                    value="",
                    lines=3,
                    interactive=False
                )
                
                record_activity_btn.click(
                    record_user_activity,
                    inputs=[activity_type_dropdown],
                    outputs=[activity_log]
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📋 待办提醒")
                reminders_state = gr.State(SIMULATED_REMINDERS.copy())
                
                def render_reminders(reminders):
                    return "\n".join([
                        f"{'✅' if r['status'] == 'confirmed' else '⏰'} {r['title']} - {r.get('medication', '')} ({r['time']})"
                        for r in reminders
                    ])
                
                reminders_display = gr.Textbox(
                    label="提醒列表",
                    value=render_reminders(SIMULATED_REMINDERS),
                    lines=6,
                    interactive=False
                )
                
                confirm_reminder_btn = gr.Button("确认已完成")
                confirm_reminder_btn.click(
                    confirm_reminder,
                    inputs=[gr.State(1), reminders_state],
                    outputs=[reminders_state]
                ).then(
                    render_reminders,
                    inputs=[reminders_state],
                    outputs=[reminders_display]
                )
            
            with gr.Column():
                gr.Markdown("## 🏠 智能家居")
                devices_state = gr.State(SIMULATED_DEVICES.copy())
                
                def render_devices(devices):
                    return "\n".join([
                        f"{'🔵' if d['status'] == 'on' else '⚪'} {d['name']} ({d['type']})"
                        for d in devices
                    ])
                
                devices_display = gr.Textbox(
                    label="设备状态",
                    value=render_devices(SIMULATED_DEVICES),
                    lines=6,
                    interactive=False
                )
                
                with gr.Row():
                    light_btn = gr.Button("客厅灯")
                    ac_btn = gr.Button("空调")
                    tv_btn = gr.Button("电视")
                
                light_btn.click(
                    toggle_device,
                    inputs=[gr.State(1), devices_state],
                    outputs=[devices_state]
                ).then(render_devices, inputs=[devices_state], outputs=[devices_display])
                
                ac_btn.click(
                    toggle_device,
                    inputs=[gr.State(2), devices_state],
                    outputs=[devices_state]
                ).then(render_devices, inputs=[devices_state], outputs=[devices_display])
                
                tv_btn.click(
                    toggle_device,
                    inputs=[gr.State(3), devices_state],
                    outputs=[devices_state]
                ).then(render_devices, inputs=[devices_state], outputs=[devices_display])
            
            with gr.Column():
                gr.Markdown("## 🏥 健康监测")
                health_state = gr.State(SIMULATED_HEALTH_DATA.copy())
                
                def render_health(data):
                    return (
                        f"🌡️ 体温: {data['temperature']}°C\n"
                        f"❤️ 心率: {data['heart_rate']} 次/分钟\n"
                        f"💨 呼吸: {data['breathing_rate']} 次/分钟\n"
                        f"💎 血氧: {data['oxygen_level']}%"
                    )
                
                health_display = gr.Textbox(
                    label="健康数据",
                    value=render_health(SIMULATED_HEALTH_DATA),
                    lines=6,
                    interactive=False
                )
                
                refresh_health_btn = gr.Button("🔄 刷新数据")
                refresh_health_btn.click(
                    update_health_data,
                    outputs=[health_state]
                ).then(render_health, inputs=[health_state], outputs=[health_display])
        
        gr.Markdown(f"⏰ 当前时间: {get_current_time()}")
    
    return demo


def execute_care_check(location: str) -> tuple:
    """
    执行智能看护检查流程
    1. 移动到指定位置
    2. 打开摄像头拍照
    3. 分析图像判断睡眠状态
    """
    logs = []
    logs.append(f"📍 收到指令：检查{location}的老人状态")
    logs.append(f"⏰ 时间：{get_current_time()}")
    logs.append("")
    
    # 步骤1：移动
    location_name = location.split(" ")[1] if " " in location else location
    logs.append("🦿 启动移动...")
    logs.append(f"正在导航到{location_name}...")
    time.sleep(0.5)
    logs.append("✅ 已到达目标位置")
    logs.append("")
    
    # 步骤2：打开摄像头
    logs.append("📷 打开摄像头...")
    if CAMERA_AVAILABLE and camera_controller.open_camera():
        logs.append("✅ 摄像头已打开")
        logs.append("")
        
        # 步骤3：拍照
        logs.append("📸 正在拍照...")
        time.sleep(0.3)
        logs.append("✅ 照片已保存")
        logs.append("")
        
        # 步骤4：分析图像
        logs.append("🔍 正在分析图像...")
        time.sleep(0.3)
        
        if CAMERA_AVAILABLE:
            result = image_analyzer.analyze_sleep_state(None)
        else:
            result = {
                "success": True,
                "is_sleeping": True,
                "eyes_closed": True,
                "is_on_bed": True,
                "confidence": 0.92
            }
        
        # 关闭摄像头
        if CAMERA_AVAILABLE:
            camera_controller.close_camera()
            logs.append("✅ 摄像头已关闭")
        
        logs.append("")
        logs.append("=" * 40)
        logs.append("📊 分析结果")
        logs.append("=" * 40)
        
        if result.get("success"):
            logs.append(f"🛏️ 是否在床上：{'是' if result.get('is_on_bed', False) else '否'}")
            logs.append(f"👁️ 眼睛状态：{'闭眼' if result.get('eyes_closed', False) else '睁眼'}")
            logs.append(f"😴 睡眠状态：{'正在睡觉' if result.get('is_sleeping', False) else '未睡觉'}")
            logs.append(f"📈 置信度：{result.get('confidence', 0):.0%}")
            
            if result.get("is_sleeping"):
                logs.append("")
                logs.append("✅ 状态正常：老人正在休息中")
            else:
                logs.append("")
                logs.append("⚠️ 提醒：老人似乎醒着")
        else:
            logs.append(f"❌ 分析失败：{result.get('error', '未知错误')}")
        
        analysis = "\n".join([
            f"🛏️ 床上状态：{'是' if result.get('is_on_bed', False) else '否'}",
            f"👁️ 眼睛：{'闭眼' if result.get('eyes_closed', False) else '睁眼'}",
            f"😴 睡眠：{'正在睡觉' if result.get('is_sleeping', False) else '未睡觉'}",
            f"📈 置信度：{result.get('confidence', 0):.0%}"
        ])
    else:
        logs.append("❌ 摄像头打开失败，使用模拟数据")
        logs.append("")
        logs.append("=" * 40)
        logs.append("📊 模拟分析结果")
        logs.append("=" * 40)
        logs.append("🛏️ 是否在床上：是")
        logs.append("👁️ 眼睛状态：闭眼")
        logs.append("😴 睡眠状态：正在睡觉")
        logs.append("📈 置信度：92%")
        logs.append("")
        logs.append("✅ 状态正常：老人正在休息中")
        
        analysis = "🛏️ 床上：是 | 👁️ 眼睛：闭眼 | 😴 睡眠：是 | 📈 置信度：92%"
    
    return "\n".join(logs), analysis

def generate_memory_summary() -> tuple:
    """生成用户习惯记忆摘要"""
    if not MEMORY_AVAILABLE:
        return "记忆模块不可用", "请确保系统正确配置"
    
    # 生成摘要
    summary = user_memory.generate_summary(user_id=1)
    
    # 格式化显示内容
    summary_text = summary.get("summary_text", "")
    details = []
    
    details.append(f"📊 分析时间段：{summary.get('time_period', '无记录')}")
    details.append(f"💬 总交互次数：{summary.get('total_interactions', 0)}")
    details.append("")
    
    # 交互类型分布
    interaction_breakdown = summary.get("interaction_breakdown", {})
    if interaction_breakdown:
        details.append("📋 交互类型分布：")
        for itype, count in interaction_breakdown.items():
            details.append(f"  • {itype}: {count}次")
        details.append("")
    
    # 活跃时间段
    active_hours = summary.get("active_hours", {})
    if active_hours:
        details.append(f"⏰ 最活跃时间：{active_hours.get('most_active_hour', '未知')}")
        details.append("")
    
    # 日常作息
    routine = summary.get("daily_routine", {})
    if routine:
        details.append("🏠 日常作息：")
        if routine.get("wake_up_time"):
            details.append(f"  • 起床：{routine.get('wake_up_time')}")
        if routine.get("sleep_time"):
            details.append(f"  • 休息：{routine.get('sleep_time')}")
        if routine.get("medicine_times"):
            details.append(f"  • 用药：{', '.join(routine.get('medicine_times', [])[:3])}")
        details.append("")
    
    # 作息建议
    suggestions = summary.get("routine_suggestions", [])
    if suggestions:
        details.append("💡 作息建议：")
        for s in suggestions[:3]:
            details.append(f"  • {s}")
    
    return summary_text, "\n".join(details)

def record_user_activity(activity_type: str) -> str:
    """记录用户活动"""
    if not MEMORY_AVAILABLE:
        return "❌ 记忆模块不可用"
    
    # 映射活动类型到内部类型
    activity_map = {
        "起床": "wake_up",
        "用餐": "meal",
        "用药": "medicine",
        "锻炼": "exercise",
        "休息": "sleep"
    }
    
    internal_type = activity_map.get(activity_type, "other")
    current_time = datetime.now().strftime("%H:%M")
    
    # 记录活动
    user_memory.record_interaction(
        interaction_type=internal_type,
        content=f"{activity_type}记录",
        user_id=1
    )
    
    return f"✅ 已记录活动：{activity_type}\n⏰ 时间：{current_time}"

if __name__ == "__main__":
    demo = create_robot_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
