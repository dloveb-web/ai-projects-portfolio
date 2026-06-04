#!/usr/bin/env python3
"""
智慧园区管理系统 - Flask版本
包含：展示前台 + 管理后台
"""

from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify
from datetime import datetime, timedelta
import random
import json

app = Flask(__name__)
app.secret_key = 'smart_park_secret_key_2026'


def generate_mock_data():
    """生成模拟数据"""
    return {
        'users': [
            {'id': 1, 'username': 'admin', 'name': '系统管理员', 'role': 'admin', 'email': 'admin@smartpark.com', 'phone': '13800138000', 'status': 'active', 'last_login': '2026-05-27 10:30'},
            {'id': 2, 'username': 'manager01', 'name': '张经理', 'role': 'manager', 'email': 'zhang@smartpark.com', 'phone': '13800138001', 'status': 'active', 'last_login': '2026-05-27 09:15'},
            {'id': 3, 'username': 'guard01', 'name': '李保安', 'role': 'guard', 'email': 'li@smartpark.com', 'phone': '13800138002', 'status': 'active', 'last_login': '2026-05-27 08:00'},
            {'id': 4, 'username': 'tech01', 'name': '王技术', 'role': 'technician', 'email': 'wang@smartpark.com', 'phone': '13800138003', 'status': 'inactive', 'last_login': '2026-05-25 17:30'},
        ],
        'parks': [
            {'id': 1, 'name': '雄安商务服务中心', 'code': 'SP-XA-001', 'address': '河北省雄安新区', 'buildings': 8, 'area': 125600, 'status': 'active'},
            {'id': 2, 'name': '中国电建科技创新产业园', 'code': 'SP-CD-001', 'address': '四川省成都市', 'buildings': 12, 'area': 258000, 'status': 'active'},
            {'id': 3, 'name': '济南华侨城', 'code': 'SP-JN-001', 'address': '山东省济南市', 'buildings': 6, 'area': 85000, 'status': 'active'},
        ],
        'devices': [
            {'id': 1, 'name': 'A栋主入口门禁', 'code': 'ACC-A001', 'type': '门禁', 'building': 'A栋', 'status': 'online', 'last_maintenance': '2026-05-01'},
            {'id': 2, 'name': 'B栋门禁控制器', 'code': 'ACC-B001', 'type': '门禁', 'building': 'B栋', 'status': 'online', 'last_maintenance': '2026-05-10'},
            {'id': 3, 'name': '园区监控摄像头01', 'code': 'CAM-001', 'type': '摄像头', 'building': '园区', 'status': 'online', 'last_maintenance': '2026-04-15'},
            {'id': 4, 'name': 'A栋温湿度传感器', 'code': 'SEN-A001', 'type': '传感器', 'building': 'A栋', 'status': 'offline', 'last_maintenance': '2026-03-20'},
            {'id': 5, 'name': '电梯1号控制器', 'code': 'ELE-001', 'type': '电梯', 'building': 'A栋', 'status': 'online', 'last_maintenance': '2026-05-15'},
        ],
        'energy': [
            {'id': 1, 'meter': 'A栋电表', 'type': '电力', 'value': 15680, 'unit': 'kWh', 'date': '2026-05-27'},
            {'id': 2, 'meter': 'B栋电表', 'type': '电力', 'value': 12890, 'unit': 'kWh', 'date': '2026-05-27'},
            {'id': 3, 'meter': '园区水表', 'type': '水', 'value': 2450, 'unit': 'm³', 'date': '2026-05-27'},
        ],
        'alerts': [
            {'id': 1, 'type': '设备告警', 'level': 'warning', 'device': 'A栋温湿度传感器', 'message': '温度异常超过35°C', 'time': '2026-05-27 10:23:45', 'status': 'pending'},
            {'id': 2, 'type': '安防告警', 'level': 'critical', 'device': '地下车库摄像头', 'message': '检测到异常入侵', 'time': '2026-05-27 09:45:12', 'status': 'processing'},
            {'id': 3, 'type': '设备告警', 'level': 'info', 'device': '电梯1号', 'message': '维保提醒', 'time': '2026-05-27 08:30:00', 'status': 'pending'},
        ],
        'access_logs': [
            {'id': 1, 'person': '张经理', 'device': 'A栋门禁', 'time': '10:30:25', 'type': '人脸', 'result': '成功'},
            {'id': 2, 'person': '李保安', 'device': '主入口门禁', 'time': '10:28:10', 'type': '刷卡', 'result': '成功'},
            {'id': 3, 'person': '访客-王五', 'device': '访客通道', 'time': '10:25:00', 'type': '二维码', 'result': '成功'},
            {'id': 4, 'person': '未知人员', 'device': 'B栋门禁', 'time': '10:20:30', 'type': '人脸', 'result': '失败'},
        ],
        'roles': [
            {'id': 1, 'name': '系统管理员', 'code': 'admin', 'permissions': ['all']},
            {'id': 2, 'name': '园区经理', 'code': 'manager', 'permissions': ['park_view', 'device_view', 'energy_view', 'report_view']},
            {'id': 3, 'name': '保安', 'code': 'guard', 'permissions': ['access_view', 'security_view', 'alert_view']},
            {'id': 4, 'name': '技术员', 'code': 'technician', 'permissions': ['device_view', 'device_config', 'alert_handle']},
        ],
    }


MOCK_DATA = generate_mock_data()


HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智慧园区综合运管平台</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; }
        .header { background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); color: white; padding: 20px 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 24px; font-weight: 600; }
        .header .time { font-size: 14px; opacity: 0.9; }
        .nav-bar { background: white; padding: 15px 30px; display: flex; gap: 20px; border-bottom: 1px solid #eee; }
        .nav-bar a { text-decoration: none; color: #666; padding: 8px 16px; border-radius: 6px; transition: all 0.3s; }
        .nav-bar a:hover, .nav-bar a.active { background: #1e3a5f; color: white; }
        .container { padding: 20px; max-width: 1400px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
        .stat-card .label { font-size: 12px; color: #666; margin-bottom: 8px; }
        .stat-card .value { font-size: 28px; font-weight: 700; color: #1e3a5f; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .panel { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .panel h2 { font-size: 16px; color: #333; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }
        .device-bar { margin-bottom: 10px; }
        .device-bar .name { font-size: 12px; color: #666; margin-bottom: 5px; }
        .device-bar .bars { display: flex; height: 20px; background: #f0f2f5; border-radius: 10px; overflow: hidden; }
        .device-bar .online { background: #28a745; }
        .device-bar .offline { background: #dc3545; }
        .chart-container { height: 200px; }
        .alert-list { list-style: none; }
        .alert-item { padding: 12px; border-left: 4px solid; margin-bottom: 10px; background: #f8f9fa; border-radius: 8px; }
        .alert-item.warning { border-color: #ffc107; }
        .alert-item.critical { border-color: #dc3545; }
        .alert-item.info { border-color: #17a2b8; }
        .alert-item .header { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .alert-item .type { font-weight: 600; }
        .alert-item .time { font-size: 12px; color: #999; }
        .alert-item .message { font-size: 13px; color: #333; }
        .access-panel { grid-column: span 3; }
        .access-table { width: 100%; border-collapse: collapse; }
        .access-table th, .access-table td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        .access-table th { background: #f5f7fa; font-weight: 600; font-size: 13px; color: #666; }
        .access-table tr:hover { background: #f8f9fa; }
        .camera-panel { grid-column: span 3; }
        .camera-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .camera-item { background: #1a1a1a; border-radius: 8px; padding: 15px; color: white; text-align: center; }
        .camera-item img { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; margin-bottom: 10px; background: #333; }
        .camera-item .name { font-size: 13px; margin-bottom: 5px; }
        .camera-item .status-tag { font-size: 12px; padding: 2px 8px; border-radius: 10px; display: inline-block; }
        .camera-item .status-tag.online { background: #28a745; }
        .camera-item .status-tag.offline { background: #dc3545; }
        .footer { display: flex; justify-content: center; gap: 15px; padding: 20px; }
        .btn { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
        .btn-primary { background: #1e3a5f; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); }
        .modal-content { background-color: white; margin: 5% auto; padding: 0; border-radius: 12px; width: 500px; max-width: 90%; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .modal-header { padding: 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .modal-header h2 { font-size: 18px; color: #333; }
        .close { color: #aaa; font-size: 28px; font-weight: bold; cursor: pointer; }
        .close:hover { color: #000; }
        .modal-body { padding: 20px; max-height: 60vh; overflow-y: auto; }
        .modal-footer { padding: 15px 20px; border-top: 1px solid #eee; display: flex; justify-content: flex-end; gap: 10px; }
        .settings-group { margin-bottom: 25px; }
        .settings-group h3 { font-size: 14px; color: #1e3a5f; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #1e3a5f; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; font-size: 13px; color: #666; margin-bottom: 5px; }
        .form-group input[type="text"], .form-group input[type="number"], .form-group input[type="date"], .form-group select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        .form-group input:focus, .form-group select:focus { outline: none; border-color: #1e3a5f; }
        .checkbox-group { display: flex; align-items: center; gap: 8px; }
        .checkbox-group input[type="checkbox"] { width: 18px; height: 18px; }
        .checkbox-group label { margin: 0; font-size: 14px; }
        @media (max-width: 768px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } .main-content { grid-template-columns: 1fr; } .camera-grid { grid-template-columns: repeat(2, 1fr); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏢 智慧园区综合运管平台</h1>
        <div class="time">{{ data.current_time }}</div>
    </div>
    <div class="nav-bar">
        <a href="/" class="active">📊 数据概览</a>
        <a href="/admin">⚙️ 管理后台</a>
    </div>
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card"><div class="label">园区楼栋</div><div class="value">{{ data.total_buildings }}栋</div></div>
            <div class="stat-card"><div class="label">设备总数</div><div class="value">{{ data.total_devices }}</div></div>
            <div class="stat-card"><div class="label">在线设备</div><div class="value">{{ data.online_devices }}</div></div>
            <div class="stat-card"><div class="label">人员总数</div><div class="value">{{ data.total_persons }}</div></div>
            <div class="stat-card"><div class="label">今日访客</div><div class="value">{{ data.today_visitors }}</div></div>
            <div class="stat-card"><div class="label">能耗(kWh)</div><div class="value">{{ data.energy_consumption }}</div></div>
        </div>
        <div class="main-content">
            <div class="panel">
                <h2>📊 设备状态分布</h2>
                <div class="device-bar"><div class="name">门禁系统</div><div class="bars"><div class="online" style="width: 96%"></div><div class="offline" style="width: 4%"></div></div></div>
                <div class="device-bar"><div class="name">监控摄像头</div><div class="bars"><div class="online" style="width: 99%"></div><div class="offline" style="width: 1%"></div></div></div>
                <div class="device-bar"><div class="name">环境传感器</div><div class="bars"><div class="online" style="width: 99%"></div><div class="offline" style="width: 1%"></div></div></div>
                <div class="device-bar"><div class="name">照明设备</div><div class="bars"><div class="online" style="width: 98%"></div><div class="offline" style="width: 2%"></div></div></div>
                <div class="device-bar"><div class="name">空调系统</div><div class="bars"><div class="online" style="width: 98%"></div><div class="offline" style="width: 2%"></div></div></div>
                <div class="device-bar"><div class="name">电梯</div><div class="bars"><div class="online" style="width: 100%"></div><div class="offline" style="width: 0%"></div></div></div>
            </div>
            <div class="panel">
                <h2>⚡ 能源消耗趋势</h2>
                <div class="chart-container"><canvas id="energyChart"></canvas></div>
            </div>
            <div class="panel">
                <h2>🚨 实时告警</h2>
                <ul class="alert-list">
                    <li class="alert-item warning"><div class="header"><span class="type">设备告警</span><span class="time">10:23:45</span></div><div class="message">A栋3楼温度异常</div></li>
                    <li class="alert-item critical"><div class="header"><span class="type">安防告警</span><span class="time">09:45:12</span></div><div class="message">地下车库异常入侵</div></li>
                    <li class="alert-item info"><div class="header"><span class="type">设备告警</span><span class="time">08:30:00</span></div><div class="message">B栋电梯维保提醒</div></li>
                </ul>
            </div>
        </div>
        <div class="main-content">
            <div class="panel access-panel">
                <h2>🚪 实时通行记录</h2>
                <table class="access-table">
                    <tr><th>时间</th><th>人员</th><th>设备</th><th>方式</th><th>状态</th><th>方向</th></tr>
                    <tr><td>10:30:25</td><td>张经理</td><td>A栋门禁</td><td>人脸</td><td>成功</td><td>进入</td></tr>
                    <tr><td>10:28:10</td><td>李保安</td><td>主入口门禁</td><td>刷卡</td><td>成功</td><td>进入</td></tr>
                    <tr><td>10:25:00</td><td>访客-王五</td><td>访客通道</td><td>二维码</td><td>成功</td><td>进入</td></tr>
                    <tr><td>10:20:30</td><td>未知人员</td><td>B栋门禁</td><td>人脸</td><td style="color:red">失败</td><td>进入</td></tr>
                </table>
            </div>
        </div>
        <div class="main-content">
            <div class="panel camera-panel">
                <h2>📷 摄像头监控</h2>
                <div class="camera-grid">
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=主入口" alt="主入口"><div class="name">CAM-001 - 主入口</div><div class="status-tag online">在线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=A栋大堂" alt="A栋大堂"><div class="name">CAM-002 - A栋大堂</div><div class="status-tag online">在线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=B栋大堂" alt="B栋大堂"><div class="name">CAM-003 - B栋大堂</div><div class="status-tag online">在线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=地下车库" alt="地下车库"><div class="name">CAM-004 - 地下车库</div><div class="status-tag online">在线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=园区广场" alt="园区广场"><div class="name">CAM-005 - 园区广场</div><div class="status-tag online">在线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=停车场" alt="停车场"><div class="name">CAM-006 - 停车场</div><div class="status-tag offline">离线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=走廊" alt="走廊"><div class="name">CAM-007 - 办公楼走廊</div><div class="status-tag online">在线</div></div>
                    <div class="camera-item"><img src="https://via.placeholder.com/320x180/333333/ffffff?text=机房" alt="机房"><div class="name">CAM-008 - 机房</div><div class="status-tag online">在线</div></div>
                </div>
            </div>
        </div>
        <div class="footer">
            <button class="btn btn-primary" onclick="location.reload()">🔄 刷新数据</button>
            <button class="btn btn-secondary" onclick="showReport()">📊 生成报表</button>
            <button class="btn btn-secondary" onclick="showSettings()">⚙️ 系统设置</button>
        </div>
    </div>
    <div id="settingsModal" class="modal"><div class="modal-content"><div class="modal-header"><h2>⚙️ 系统设置</h2><span class="close" onclick="closeSettings()">&times;</span></div><div class="modal-body"><div class="settings-group"><h3>园区配置</h3><div class="form-group"><label>园区名称</label><input type="text" id="parkName" value="智慧园区综合运管平台"></div><div class="form-group"><label>园区编号</label><input type="text" id="parkCode" value="SP-2026-001"></div><div class="form-group"><label>所在城市</label><input type="text" id="city" value="北京市"></div></div><div class="settings-group"><h3>通知设置</h3><div class="form-group checkbox-group"><input type="checkbox" id="emailNotify" checked><label>启用邮件通知</label></div><div class="form-group checkbox-group"><input type="checkbox" id="smsNotify"><label>启用短信通知</label></div><div class="form-group checkbox-group"><input type="checkbox" id="wechatNotify" checked><label>启用微信推送</label></div></div><div class="settings-group"><h3>告警阈值</h3><div class="form-group"><label>温度告警阈值 (°C)</label><input type="number" id="tempThreshold" value="35"></div><div class="form-group"><label>能耗告警阈值 (kWh)</label><input type="number" id="energyThreshold" value="10000"></div></div><div class="settings-group"><h3>数据刷新</h3><div class="form-group"><label>自动刷新间隔</label><select id="refreshInterval"><option value="5">5秒</option><option value="10" selected>10秒</option><option value="30">30秒</option><option value="60">1分钟</option><option value="0">禁用</option></select></div></div></div><div class="modal-footer"><button class="btn btn-secondary" onclick="closeSettings()">取消</button><button class="btn btn-primary" onclick="saveSettings()">保存设置</button></div></div></div>
    <div id="reportModal" class="modal"><div class="modal-content"><div class="modal-header"><h2>📊 生成报表</h2><span class="close" onclick="closeReport()">&times;</span></div><div class="modal-body"><div class="form-group"><label>报表类型</label><select id="reportType"><option value="daily">日报</option><option value="weekly">周报</option><option value="monthly">月报</option><option value="yearly">年报</option></select></div><div class="form-group"><label>日期范围</label><input type="date" id="reportDate"></div><div class="form-group"><label>报表格式</label><select id="reportFormat"><option value="pdf">PDF</option><option value="excel">Excel</option><option value="word">Word</option></select></div><div class="form-group"><label>包含模块</label><div class="checkbox-group"><input type="checkbox" id="modPark" checked><label>园区概览</label></div><div class="checkbox-group"><input type="checkbox" id="modDevice" checked><label>设备状态</label></div><div class="checkbox-group"><input type="checkbox" id="modEnergy" checked><label>能源消耗</label></div><div class="checkbox-group"><input type="checkbox" id="modAccess" checked><label>通行记录</label></div><div class="checkbox-group"><input type="checkbox" id="modSecurity"><label>安防告警</label></div></div></div><div class="modal-footer"><button class="btn btn-secondary" onclick="closeReport()">取消</button><button class="btn btn-primary" onclick="generateReport()">生成报表</button></div></div></div>
    <script>var ctx = document.getElementById('energyChart').getContext('2d');var chart = new Chart(ctx, {type: 'line', data: {labels: ['00:00','02:00','04:00','06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00'], datasets: [{label: '电力(kWh)', data: [320,280,260,290,450,680,720,690,580,520,420,360], borderColor: '#28a745', tension: 0.3, fill: false}, {label: '用水(吨)', data: [65,58,55,62,85,110,118,105,92,78,68,62], borderColor: '#17a2b8', tension: 0.3, fill: false}]}, options: {responsive: true, maintainAspectRatio: false, scales: {y: {beginAtZero: true}}}}); function showSettings(){document.getElementById('settingsModal').style.display='block';} function closeSettings(){document.getElementById('settingsModal').style.display='none';} function saveSettings(){alert('设置已保存！'); closeSettings();} function showReport(){document.getElementById('reportModal').style.display='block';} function closeReport(){document.getElementById('reportModal').style.display='none';} function generateReport(){alert('报表生成中...'); closeReport();} window.onclick=function(event){if(event.target==document.getElementById('settingsModal')){document.getElementById('settingsModal').style.display='none';} if(event.target==document.getElementById('reportModal')){document.getElementById('reportModal').style.display='none';}}</script>
</body>
</html>
"""


HTML_ADMIN = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智慧园区 - 管理后台</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; }
        .header { background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 20px; }
        .header a { color: white; text-decoration: none; margin-left: 20px; }
        .container { display: flex; min-height: calc(100vh - 60px); }
        .sidebar { width: 220px; background: white; border-right: 1px solid #e0e0e0; padding: 20px 0; }
        .sidebar-menu { list-style: none; }
        .sidebar-menu li { margin-bottom: 5px; }
        .sidebar-menu a { display: block; padding: 12px 20px; color: #333; text-decoration: none; transition: all 0.3s; border-left: 3px solid transparent; }
        .sidebar-menu a:hover, .sidebar-menu a.active { background: #f0f4f8; border-left-color: #1e3a5f; color: #1e3a5f; }
        .sidebar-menu a .icon { margin-right: 10px; }
        .content { flex: 1; padding: 20px 30px; }
        .page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .page-header h2 { font-size: 20px; color: #333; }
        .btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; margin-left: 10px; }
        .btn-primary { background: #1e3a5f; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: #333; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
        .card-header h3 { font-size: 16px; color: #333; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; font-size: 13px; color: #666; }
        tr:hover { background: #f8f9fa; }
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-info { background: #d1ecf1; color: #0c5460; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
        .status-dot.online { background: #28a745; }
        .status-dot.offline { background: #dc3545; }
        .pagination { display: flex; justify-content: center; align-items: center; margin-top: 20px; gap: 5px; }
        .pagination button { padding: 6px 12px; border: 1px solid #ddd; background: white; cursor: pointer; border-radius: 4px; }
        .pagination button:hover { background: #f0f0f0; }
        .pagination button.active { background: #1e3a5f; color: white; border-color: #1e3a5f; }
        .search-box { display: flex; gap: 10px; margin-bottom: 15px; }
        .search-box input { flex: 1; padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; }
        .search-box select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; }
        .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-item { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .stat-item .number { font-size: 28px; font-weight: 700; color: #1e3a5f; }
        .stat-item .label { font-size: 13px; color: #666; margin-top: 5px; }
        .stat-item .icon { font-size: 30px; margin-bottom: 10px; }
        .tab-nav { display: flex; border-bottom: 2px solid #eee; margin-bottom: 20px; }
        .tab-nav button { padding: 12px 24px; border: none; background: none; cursor: pointer; font-size: 14px; color: #666; border-bottom: 2px solid transparent; margin-bottom: -2px; }
        .tab-nav button:hover, .tab-nav button.active { color: #1e3a5f; border-bottom-color: #1e3a5f; }
        .form-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; font-size: 13px; color: #666; margin-bottom: 5px; }
        .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        .form-group textarea { min-height: 80px; resize: vertical; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); }
        .modal-content { background-color: white; margin: 5% auto; padding: 0; border-radius: 12px; width: 600px; max-width: 90%; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .modal-header { padding: 15px 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .modal-header h2 { font-size: 18px; }
        .close { font-size: 28px; cursor: pointer; color: #999; }
        .close:hover { color: #333; }
        .modal-body { padding: 20px; max-height: 60vh; overflow-y: auto; }
        .modal-footer { padding: 15px 20px; border-top: 1px solid #eee; display: flex; justify-content: flex-end; gap: 10px; }
        .actions button { padding: 5px 10px; margin-right: 5px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .actions .view { background: #17a2b8; color: white; }
        .actions .edit { background: #ffc107; color: #333; }
        .actions .delete { background: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚙️ 管理后台</h1>
        <div>
            <a href="/">📊 返回首页</a>
            <span>欢迎，系统管理员</span>
        </div>
    </div>
    <div class="container">
        <div class="sidebar">
            <ul class="sidebar-menu">
                <li><a href="#" class="active" onclick="showSection('dashboard')"><span class="icon">📊</span>工作台概览</a></li>
                <li><a href="#" onclick="showSection('users')"><span class="icon">👥</span>用户管理</a></li>
                <li><a href="#" onclick="showSection('parks')"><span class="icon">🏢</span>园区管理</a></li>
                <li><a href="#" onclick="showSection('devices')"><span class="icon">📱</span>设备管理</a></li>
                <li><a href="#" onclick="showSection('energy')"><span class="icon">⚡</span>能源管理</a></li>
                <li><a href="#" onclick="showSection('access')"><span class="icon">🚪</span>通行管理</a></li>
                <li><a href="#" onclick="showSection('security')"><span class="icon">🔒</span>安防管理</a></li>
                <li><a href="#" onclick="showSection('roles')"><span class="icon">🔐</span>权限管理</a></li>
                <li><a href="#" onclick="showSection('logs')"><span class="icon">📝</span>操作日志</a></li>
                <li><a href="#" onclick="showSection('settings')"><span class="icon">⚙️</span>系统设置</a></li>
            </ul>
        </div>
        <div class="content">
            <div id="section-dashboard" class="section">
                <div class="page-header"><h2>📊 工作台概览</h2></div>
                <div class="stats-row">
                    <div class="stat-item"><div class="icon">👥</div><div class="number">{{ stats.users }}</div><div class="label">用户总数</div></div>
                    <div class="stat-item"><div class="icon">🏢</div><div class="number">{{ stats.parks }}</div><div class="label">园区数量</div></div>
                    <div class="stat-item"><div class="icon">📱</div><div class="number">{{ stats.devices }}</div><div class="label">设备总数</div></div>
                    <div class="stat-item"><div class="icon">🚨</div><div class="number">{{ stats.alerts }}</div><div class="label">待处理告警</div></div>
                </div>
                <div class="card">
                    <div class="card-header"><h3>📋 快捷操作</h3></div>
                    <button class="btn btn-primary" onclick="showSection('users')">添加用户</button>
                    <button class="btn btn-success" onclick="showSection('parks')">添加园区</button>
                    <button class="btn btn-warning" onclick="showSection('devices')">添加设备</button>
                </div>
                <div class="card">
                    <div class="card-header"><h3>🚨 待处理告警</h3></div>
                    <table>
                        <tr><th>告警类型</th><th>级别</th><th>设备</th><th>消息</th><th>时间</th><th>状态</th><th>操作</th></tr>
                        {% for alert in alerts %}
                        <tr>
                            <td>{{ alert.type }}</td>
                            <td><span class="badge badge-{{ 'danger' if alert.level=='critical' else 'warning' if alert.level=='warning' else 'info' }}">{{ alert.level }}</span></td>
                            <td>{{ alert.device }}</td>
                            <td>{{ alert.message }}</td>
                            <td>{{ alert.time }}</td>
                            <td><span class="badge badge-{{ 'success' if alert.status=='resolved' else 'warning' }}">{{ alert.status }}</span></td>
                            <td class="actions"><button class="view">查看</button><button class="edit">处理</button></td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            <div id="section-users" class="section" style="display:none">
                <div class="page-header"><h2>👥 用户管理</h2><button class="btn btn-primary" onclick="openModal('userModal')">+ 添加用户</button></div>
                <div class="card">
                    <div class="search-box">
                        <input type="text" placeholder="搜索用户名、姓名...">
                        <select><option>全部角色</option><option>管理员</option><option>经理</option><option>保安</option><option>技术员</option></select>
                        <select><option>全部状态</option><option>启用</option><option>禁用</option></select>
                        <button class="btn btn-primary">搜索</button>
                    </div>
                    <table>
                        <tr><th>ID</th><th>用户名</th><th>姓名</th><th>角色</th><th>邮箱</th><th>电话</th><th>最后登录</th><th>状态</th><th>操作</th></tr>
                        {% for user in users %}
                        <tr>
                            <td>{{ user.id }}</td>
                            <td>{{ user.username }}</td>
                            <td>{{ user.name }}</td>
                            <td><span class="badge badge-info">{{ user.role }}</span></td>
                            <td>{{ user.email }}</td>
                            <td>{{ user.phone }}</td>
                            <td>{{ user.last_login }}</td>
                            <td><span class="badge badge-{{ 'success' if user.status=='active' else 'danger' }}">{{ user.status }}</span></td>
                            <td class="actions"><button class="view">查看</button><button class="edit">编辑</button><button class="delete">删除</button></td>
                        </tr>
                        {% endfor %}
                    </table>
                    <div class="pagination">
                        <button>&laquo;</button><button class="active">1</button><button>2</button><button>3</button><button>&raquo;</button>
                    </div>
                </div>
            </div>
            <div id="section-parks" class="section" style="display:none">
                <div class="page-header"><h2>🏢 园区管理</h2><button class="btn btn-primary">+ 添加园区</button></div>
                <div class="card">
                    <table>
                        <tr><th>ID</th><th>园区名称</th><th>编码</th><th>地址</th><th>楼栋数</th><th>面积(m²)</th><th>状态</th><th>操作</th></tr>
                        {% for park in parks %}
                        <tr>
                            <td>{{ park.id }}</td>
                            <td>{{ park.name }}</td>
                            <td>{{ park.code }}</td>
                            <td>{{ park.address }}</td>
                            <td>{{ park.buildings }}</td>
                            <td>{{ park.area }}</td>
                            <td><span class="badge badge-success">{{ park.status }}</span></td>
                            <td class="actions"><button class="view">查看</button><button class="edit">编辑</button></td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            <div id="section-devices" class="section" style="display:none">
                <div class="page-header"><h2>📱 设备管理</h2><button class="btn btn-primary">+ 添加设备</button></div>
                <div class="card">
                    <table>
                        <tr><th>ID</th><th>设备名称</th><th>编码</th><th>类型</th><th>位置</th><th>状态</th><th>最后维护</th><th>操作</th></tr>
                        {% for device in devices %}
                        <tr>
                            <td>{{ device.id }}</td>
                            <td>{{ device.name }}</td>
                            <td>{{ device.code }}</td>
                            <td>{{ device.type }}</td>
                            <td>{{ device.building }}</td>
                            <td><span class="status-dot {{ device.status }}"></span>{{ device.status }}</td>
                            <td>{{ device.last_maintenance }}</td>
                            <td class="actions"><button class="view">查看</button><button class="edit">配置</button></td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            <div id="section-energy" class="section" style="display:none">
                <div class="page-header"><h2>⚡ 能源管理</h2><button class="btn btn-primary">查看报表</button></div>
                <div class="stats-row">
                    <div class="stat-item"><div class="number">28,570</div><div class="label">今日用电(kWh)</div></div>
                    <div class="stat-item"><div class="number">2,450</div><div class="label">今日用水(m³)</div></div>
                    <div class="stat-item"><div class="number">15,680</div><div class="label">A栋用电(kWh)</div></div>
                    <div class="stat-item"><div class="number">12,890</div><div class="label">B栋用电(kWh)</div></div>
                </div>
                <div class="card">
                    <div class="card-header"><h3>📊 实时能耗数据</h3></div>
                    <table>
                        <tr><th>计量表</th><th>类型</th><th>当前值</th><th>单位</th><th>日期</th></tr>
                        {% for e in energy %}
                        <tr><td>{{ e.meter }}</td><td>{{ e.type }}</td><td>{{ e.value }}</td><td>{{ e.unit }}</td><td>{{ e.date }}</td></tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            <div id="section-access" class="section" style="display:none">
                <div class="page-header"><h2>🚪 通行管理</h2><button class="btn btn-primary">通行记录导出</button></div>
                <div class="card">
                    <div class="search-box">
                        <input type="text" placeholder="搜索人员姓名...">
                        <select><option>全部类型</option><option>人脸</option><option>刷卡</option><option>二维码</option><option>指纹</option></select>
                        <select><option>全部结果</option><option>成功</option><option>失败</option></select>
                        <button class="btn btn-primary">搜索</button>
                    </div>
                    <table>
                        <tr><th>ID</th><th>人员</th><th>设备</th><th>时间</th><th>方式</th><th>结果</th></tr>
                        {% for log in access_logs %}
                        <tr><td>{{ log.id }}</td><td>{{ log.person }}</td><td>{{ log.device }}</td><td>{{ log.time }}</td><td>{{ log.type }}</td><td><span class="badge badge-{{ 'success' if log.result=='成功' else 'danger' }}">{{ log.result }}</span></td></tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            <div id="section-security" class="section" style="display:none">
                <div class="page-header"><h2>🔒 安防管理</h2><button class="btn btn-primary">监控大屏</button></div>
                <div class="card">
                    <div class="tab-nav">
                        <button class="active">摄像头管理</button>
                        <button>告警记录</button>
                        <button>巡更管理</button>
                    </div>
                    <table>
                        <tr><th>摄像头ID</th><th>位置</th><th>分辨率</th><th>状态</th><th>最后在线</th><th>操作</th></tr>
                        <tr><td>CAM-001</td><td>主入口</td><td>1080P</td><td><span class="status-dot online"></span>在线</td><td>2026-05-27 10:30</td><td><button class="edit">配置</button></td></tr>
                        <tr><td>CAM-002</td><td>A栋大堂</td><td>1080P</td><td><span class="status-dot online"></span>在线</td><td>2026-05-27 10:30</td><td><button class="edit">配置</button></td></tr>
                        <tr><td>CAM-003</td><td>B栋大堂</td><td>1080P</td><td><span class="status-dot online"></span>在线</td><td>2026-05-27 10:30</td><td><button class="edit">配置</button></td></tr>
                        <tr><td>CAM-004</td><td>地下车库</td><td>720P</td><td><span class="status-dot offline"></span>离线</td><td>2026-05-27 08:15</td><td><button class="edit">配置</button></td></tr>
                    </table>
                </div>
            </div>
            <div id="section-roles" class="section" style="display:none">
                <div class="page-header"><h2>🔐 权限管理</h2><button class="btn btn-primary">+ 添加角色</button></div>
                <div class="card">
                    <table>
                        <tr><th>ID</th><th>角色名称</th><th>编码</th><th>权限</th><th>操作</th></tr>
                        {% for role in roles %}
                        <tr>
                            <td>{{ role.id }}</td>
                            <td>{{ role.name }}</td>
                            <td>{{ role.code }}</td>
                            <td><span class="badge badge-info">{{ role.permissions|length }} 个权限</span></td>
                            <td><button class="edit">编辑</button></td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            <div id="section-logs" class="section" style="display:none">
                <div class="page-header"><h2>📝 操作日志</h2><button class="btn btn-primary">导出日志</button></div>
                <div class="card">
                    <table>
                        <tr><th>时间</th><th>用户</th><th>操作</th><th>模块</th><th>IP地址</th></tr>
                        <tr><td>2026-05-27 10:30:25</td><td>admin</td><td>登录系统</td><td>认证</td><td>192.168.1.100</td></tr>
                        <tr><td>2026-05-27 10:25:10</td><td>admin</td><td>添加设备</td><td>设备管理</td><td>192.168.1.100</td></tr>
                        <tr><td>2026-05-27 09:15:30</td><td>manager01</td><td>登录系统</td><td>认证</td><td>192.168.1.101</td></tr>
                    </table>
                </div>
            </div>
            <div id="section-settings" class="section" style="display:none">
                <div class="page-header"><h2>⚙️ 系统设置</h2></div>
                <div class="card">
                    <div class="card-header"><h3>基本设置</h3></div>
                    <div class="form-row">
                        <div class="form-group"><label>系统名称</label><input type="text" value="智慧园区综合运管平台"></div>
                        <div class="form-group"><label>系统版本</label><input type="text" value="v1.0.0" disabled></div>
                    </div>
                    <div class="form-group"><label>系统描述</label><textarea>基于太极股份智慧园区解决方案的现代化园区管理系统</textarea></div>
                </div>
                <div class="card">
                    <div class="card-header"><h3>通知设置</h3></div>
                    <div class="form-group checkbox"><input type="checkbox" checked> 启用邮件通知</div>
                    <div class="form-group checkbox"><input type="checkbox"> 启用短信通知</div>
                    <div class="form-group checkbox"><input type="checkbox" checked> 启用微信推送</div>
                </div>
                <div class="card">
                    <div class="card-header"><h3>告警阈值</h3></div>
                    <div class="form-row">
                        <div class="form-group"><label>温度告警阈值 (°C)</label><input type="number" value="35"></div>
                        <div class="form-group"><label>能耗告警阈值 (kWh)</label><input type="number" value="10000"></div>
                    </div>
                </div>
                <button class="btn btn-primary">保存设置</button>
            </div>
        </div>
    </div>
    
    <div id="userModal" class="modal">
        <div class="modal-content">
            <div class="modal-header"><h2>添加用户</h2><span class="close" onclick="closeModal('userModal')">&times;</span></div>
            <div class="modal-body">
                <div class="form-row">
                    <div class="form-group"><label>用户名</label><input type="text" placeholder="请输入用户名"></div>
                    <div class="form-group"><label>姓名</label><input type="text" placeholder="请输入姓名"></div>
                </div>
                <div class="form-row">
                    <div class="form-group"><label>角色</label><select><option>系统管理员</option><option>园区经理</option><option>保安</option><option>技术员</option></select></div>
                    <div class="form-group"><label>电话</label><input type="text" placeholder="请输入电话"></div>
                </div>
                <div class="form-group"><label>邮箱</label><input type="email" placeholder="请输入邮箱"></div>
                <div class="form-group"><label>初始密码</label><input type="password" placeholder="请输入密码"></div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal('userModal')">取消</button>
                <button class="btn btn-primary">保存</button>
            </div>
        </div>
    </div>
    
    <script>
        function showSection(name) {
            document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
            document.getElementById('section-' + name).style.display = 'block';
            document.querySelectorAll('.sidebar-menu a').forEach(a => a.classList.remove('active'));
            event.target.classList.add('active');
        }
        function openModal(id) { document.getElementById(id).style.display = 'block'; }
        function closeModal(id) { document.getElementById(id).style.display = 'none'; }
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """数据概览页面"""
    data = {
        'current_time': datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
        'total_buildings': 8,
        'total_devices': 1256,
        'online_devices': 1234,
        'total_persons': 2856,
        'today_visitors': 128,
        'energy_consumption': 15680,
    }
    return render_template_string(HTML_DASHBOARD, data=data)


@app.route('/admin')
def admin():
    """管理后台页面"""
    data = {
        'stats': {
            'users': len(MOCK_DATA['users']),
            'parks': len(MOCK_DATA['parks']),
            'devices': len(MOCK_DATA['devices']),
            'alerts': len([a for a in MOCK_DATA['alerts'] if a['status'] == 'pending']),
        },
        'users': MOCK_DATA['users'],
        'parks': MOCK_DATA['parks'],
        'devices': MOCK_DATA['devices'],
        'energy': MOCK_DATA['energy'],
        'alerts': MOCK_DATA['alerts'],
        'access_logs': MOCK_DATA['access_logs'],
        'roles': MOCK_DATA['roles'],
    }
    return render_template_string(HTML_ADMIN, **data)


@app.route('/api/users')
def api_users():
    """用户API"""
    return jsonify({'code': 200, 'data': MOCK_DATA['users']})


@app.route('/api/parks')
def api_parks():
    """园区API"""
    return jsonify({'code': 200, 'data': MOCK_DATA['parks']})


@app.route('/api/devices')
def api_devices():
    """设备API"""
    return jsonify({'code': 200, 'data': MOCK_DATA['devices']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)
