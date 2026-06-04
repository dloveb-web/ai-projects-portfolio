#!/usr/bin/env python3
"""
WebSocket连接管理
"""

from typing import Dict, Set
from fastapi import WebSocket


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, room_id: str, websocket: WebSocket):
        """连接到房间"""
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()
        self.active_connections[room_id].add(websocket)
    
    def disconnect(self, room_id: str, websocket: WebSocket):
        """断开连接"""
        if room_id in self.active_connections:
            self.active_connections[room_id].discard(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        await websocket.send_text(message)
    
    async def broadcast(self, room_id: str, message: str):
        """广播消息到房间"""
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)


manager = ConnectionManager()
