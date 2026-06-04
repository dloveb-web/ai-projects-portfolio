#!/usr/bin/env python3
"""
智慧园区前端展示页面
"""

import gradio as gr
from datetime import datetime, timedelta
import random
from typing import Dict, Any


# 模拟数据生成
def generate_park_overview() -> Dict[str, Any]:
    """生成园区概览数据"""
    return {
        "park_name": "智慧园区综合运管平台",
        "total_buildings": 8,
        "total_devices": 1256,
        "online_devices": 1234,
        "total_persons": 2856,
        "today_visitors": 128,
        "energy_consumption": 15680,
        "alert_count": 3,
        "security_status": "正常",
    }


def generate_device_stats() -> Dict[str, Any]:
    """生成设备统计数据"""
    categories = ["门禁系统", "监控摄像头", "环境传感器", "照明设备", "空调系统", "电梯"]
    return {
        "categories": categories,
        "online": [45, 128, 356, 520, 86, 12],
        "offline": [2, 1, 5, 8, 2, 0],
        "total": [47, 129, 361, 528, 88, 12],
    }


def generate_energy_data() -> Dict[str, Any]:
    """生成能源数据"""
    hours = [f"{i:02d}:00" for i in range(24)]
    electricity = [random.randint(400, 800) for _ in range(24)]
    water = [random.randint(80, 150) for _ in range(24)]
    return {
        "hours": hours,
        "electricity": electricity,
        "water": water,
    }


def generate_access_records() -> list:
    """生成通行记录"""
    records = []
    persons = ["张三", "李四", "王五", "赵六", "钱七"]
    devices = ["主入口门禁", "A栋门禁", "B栋门禁", "地下车库", "员工通道"]
    types = ["刷卡", "人脸", "二维码", "指纹"]
    now = datetime.now()
    for _ in range(10):
        records.append({
            "time": (now - timedelta(minutes=random.randint(1, 120))).strftime("%H:%M:%S"),
            "person": random.choice(persons),
            "device": random.choice(devices),
            "type": random.choice(types),
            "status": "成功" if random.random() > 0.05 else "失败",
            "direction": random.choice(["进入", "离开"]),
        })
    return records


def generate_alerts() -> list:
    """生成告警列表"""
    return [
        {"id": 1, "type": "设备告警", "level": "警告", "message": "A栋3楼温度异常", "time": "10:23:45", "status": "待处理"},
        {"id": 2, "type": "安防告警", "level": "严重", "message": "地下车库异常入侵", "time": "09:45:12", "status": "处理中"},
        {"id": 3, "type": "设备告警", "level": "信息", "message": "B栋电梯维保提醒", "time": "08:30:00", "status": "待处理"},
    ]


def generate_security_cameras() -> list:
    """生成摄像头列表"""
    locations = ["主入口", "A栋大堂", "B栋大堂", "地下车库", "园区广场", "停车场出入口", "办公楼走廊", "机房"]
    return [
        {"id": f"CAM{i:03d}", "location": locations[i], "status": random.choice(["在线", "离线"]) if i != 3 else "在线", "resolution": "1080P"}
        for i in range(8)
    ]


# Gradio界面布局
with gr.Blocks(title="智慧园区综合运管平台", theme=gr.themes.Default()) as demo:
    # 顶部标题栏
    with gr.Row(elem_classes=["header"]):
        gr.Markdown("# 🏢 智慧园区综合运管平台", elem_classes=["title"])
        gr.Markdown(f"**{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}**", elem_classes=["time"])
    
    # 园区概览卡片
    with gr.Row():
        stats = generate_park_overview()
        metrics = [
            ("园区楼栋", stats["total_buildings"], "栋"),
            ("设备总数", stats["total_devices"], "台"),
            ("在线设备", stats["online_devices"], "台"),
            ("人员总数", stats["total_persons"], "人"),
            ("今日访客", stats["today_visitors"], "人"),
            ("能耗(kWh)", stats["energy_consumption"], ""),
        ]
        for label, value, unit in metrics:
            with gr.Column():
                gr.Card(
                    gr.Markdown(f"## {value}{unit}"),
                    title=label,
                    elem_classes=["stat-card"]
                )
    
    # 安全状态指示
    with gr.Row():
        status_color = "green" if stats["security_status"] == "正常" else "red"
        gr.Card(
            gr.Markdown(f"**安防状态:** <span style='color:{status_color};font-size:24px'>{stats['security_status']}</span>"),
            title="系统状态",
            elem_classes=["status-card"]
        )
        gr.Card(
            gr.Markdown(f"**待处理告警:** <span style='color:red;font-size:24px'>{stats['alert_count']}</span> 条"),
            title="告警统计",
            elem_classes=["alert-card"]
        )
    
    # 主内容区
    with gr.Row():
        # 左侧：设备状态
        with gr.Column(scale=1):
            gr.Markdown("## 📊 设备状态分布")
            device_stats = generate_device_stats()
            with gr.Group():
                for i, cat in enumerate(device_stats["categories"]):
                    gr.BarChart(
                        [[{"name": "在线", "value": device_stats["online"][i]}, {"name": "离线", "value": device_stats["offline"][i]}]],
                        title=cat,
                        x_labels=["状态"],
                        elem_classes=["mini-chart"]
                    )
        
        # 中间：能源消耗趋势
        with gr.Column(scale=1):
            gr.Markdown("## ⚡ 能源消耗趋势")
            energy_data = generate_energy_data()
            gr.LineChart(
                [{"name": "电力(kWh)", "data": energy_data["electricity"]}, {"name": "用水(吨)", "data": energy_data["water"]}],
                x_labels=energy_data["hours"],
                title="24小时能耗",
                elem_classes=["main-chart"]
            )
        
        # 右侧：实时告警
        with gr.Column(scale=1):
            gr.Markdown("## 🚨 实时告警")
            alerts = generate_alerts()
            alert_table = gr.DataFrame(
                alerts,
                headers=["ID", "类型", "级别", "消息", "时间", "状态"],
                elem_classes=["alert-table"]
            )
            gr.Button("处理告警", elem_classes=["action-btn"])
    
    # 通行记录
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🚪 实时通行记录")
            access_df = gr.DataFrame(
                generate_access_records(),
                headers=["时间", "人员", "设备", "方式", "状态", "方向"],
                elem_classes=["access-table"]
            )
    
    # 摄像头监控
    with gr.Row():
        gr.Markdown("## 📷 摄像头监控")
        cameras = generate_security_cameras()
        camera_grid = gr.GridGallery(
            [f"http://placehold.it/320x180?text={cam['location']}" for cam in cameras],
            captions=[f"{cam['id']} - {cam['location']} ({cam['status']})" for cam in cameras],
            columns=4,
            elem_classes=["camera-grid"]
        )
    
    # 底部操作区
    with gr.Row(elem_classes=["footer"]):
        gr.Button("🔄 刷新数据", elem_classes=["refresh-btn"]).click(
            fn=lambda: generate_park_overview(),
            outputs=[]
        )
        gr.Button("📊 生成报表", elem_classes=["report-btn"])
        gr.Button("⚙️ 系统设置", elem_classes=["settings-btn"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
