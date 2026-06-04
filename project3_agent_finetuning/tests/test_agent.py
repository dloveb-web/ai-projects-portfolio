import pytest
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent import agent_manager


class TestAgentManager:
    """Agent管理器测试套件"""
    
    def test_initialization(self):
        """测试Agent管理器初始化"""
        assert agent_manager is not None
        assert hasattr(agent_manager, 'sessions')
        assert isinstance(agent_manager.sessions, dict)
    
    def test_get_available_agents(self):
        """测试获取可用Agent类型"""
        types = agent_manager.get_available_agents()
        assert isinstance(types, list)
        assert len(types) >= 3
        assert "react" in types
        assert "multiagent" in types
        assert "visual" in types
    
    def test_create_agent_session(self):
        """测试创建Agent会话"""
        session = agent_manager.create_session(
            session_id=997,
            agent_type="react",
            config={}
        )
        assert session['id'] == 997
        assert session['agent_type'] == "react"
        assert 'agent' in session
    
    def test_react_agent_mock_mode(self):
        """测试ReAct Agent的模拟模式"""
        session = agent_manager.create_session(
            session_id=996,
            agent_type="react",
            config={}
        )
        agent = session['agent']
        assert hasattr(agent, 'mock_mode')
        assert isinstance(agent.mock_mode, bool)
