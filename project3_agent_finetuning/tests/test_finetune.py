import pytest
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.finetune import finetune_manager


class TestFinetuneManager:
    """微调管理器测试套件"""
    
    def test_initialization(self):
        """测试微调管理器初始化"""
        assert finetune_manager is not None
        assert hasattr(finetune_manager, 'tasks')
        assert finetune_manager.output_base_path.exists()
        assert hasattr(finetune_manager, 'mock_mode')
    
    def test_create_task(self):
        """测试创建微调任务"""
        task = finetune_manager.create_task(
            task_id=998,
            model_name="test-model",
            method="qlora",
            config={"lr": 0.001}
        )
        assert task['id'] == 998
        assert task['model_name'] == "test-model"
        assert task['method'] == "qlora"
        assert task['status'] == "pending"
        assert 'is_mock' in task
    
    def test_get_available_methods(self):
        """测试获取可用方法"""
        methods = finetune_manager.get_available_methods()
        assert isinstance(methods, list)
        assert len(methods) >= 4
        assert "qlora" in methods
        assert "lora" in methods
        assert "adapter" in methods
        assert "dpo" in methods
    
    def test_mock_mode_flag(self):
        """测试模拟模式标志"""
        assert isinstance(finetune_manager.mock_mode, bool)
