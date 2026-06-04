"""
摄像头控制和图像分析模块
用于智能看护场景：检查老人睡眠状态
"""
import cv2
import base64
from PIL import Image
from io import BytesIO
from typing import Optional, Dict
import random
import os

try:
    import dashscope
    from dashscope.multimodal import MultiModalConversation
except ImportError:
    dashscope = None


class CameraController:
    """摄像头控制器"""
    
    def __init__(self):
        self.camera = None
        self.is_opened = False
        self.last_frame = None
        
    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.is_opened = True
                return True
            else:
                return False
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            return False
    
    def capture_frame(self) -> Optional[str]:
        """捕获一帧图像，返回base64编码"""
        if not self.is_opened or self.camera is None:
            return None
            
        ret, frame = self.camera.read()
        if ret:
            self.last_frame = frame
            # 转换为base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return img_str
        return None
    
    def close_camera(self):
        """关闭摄像头"""
        if self.camera:
            self.camera.release()
            self.is_opened = False


class ImageAnalyzer:
    """图像分析器 - 使用阿里百炼视觉模型"""
    
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if self.api_key and dashscope:
            dashscope.api_key = self.api_key
    
    def analyze_sleep_state(self, image_base64: str) -> Dict:
        """
        分析睡眠状态
        返回：是否在睡觉、眼睛状态、置信度等
        """
        if not self.api_key or not dashscope:
            return self._simulate_analysis()
        
        try:
            # 调用阿里百炼视觉模型
            response = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{image_base64}",
                            "text": "请分析这张图片中的人物状态：\n"
                                   "1. 这个人是否在床上？\n"
                                   "2. 眼睛是睁开还是闭着？\n"
                                   "3. 是否在睡觉？请给出判断和置信度。\n"
                                   "请用JSON格式返回结果。"
                        }
                    ]
                }]
            )
            
            if response.status_code == 200:
                return self._parse_vision_response(response)
            else:
                return {
                    "success": False,
                    "error": f"API调用失败: {response.message}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"分析异常: {str(e)}"
            }
    
    def _parse_vision_response(self, response) -> Dict:
        """解析视觉模型响应"""
        try:
            content = response.output.choices[0].message.content
            # 这里简化处理，实际应该解析JSON
            return {
                "success": True,
                "raw_response": content,
                "summary": "图像分析完成"
            }
        except:
            return {
                "success": False,
                "error": "响应解析失败"
            }
    
    def _simulate_analysis(self) -> Dict:
        """模拟分析结果（当API不可用时）"""
        # 模拟不同的分析结果
        states = [
            {"is_on_bed": True, "eyes_closed": True, "is_sleeping": True, "confidence": 0.95},
            {"is_on_bed": True, "eyes_closed": True, "is_sleeping": True, "confidence": 0.92},
            {"is_on_bed": True, "eyes_closed": False, "is_sleeping": False, "confidence": 0.88},
        ]
        
        result = random.choice(states)
        return {
            "success": True,
            "is_on_bed": result["is_on_bed"],
            "eyes_closed": result["eyes_closed"],
            "is_sleeping": result["is_sleeping"],
            "confidence": result["confidence"],
            "note": "模拟分析结果（需配置阿里百炼视觉API）"
        }


def simulate_robot_movement(target_location: str) -> str:
    """模拟机器人移动到指定位置"""
    movements = [
        f"正在移动到{target_location}...",
        f"导航中，目标：{target_location}",
        "检测到障碍物，绕行中...",
        "已到达目标位置",
        "正在调整角度..."
    ]
    return "\n".join(movements)


# 全局实例
camera_controller = CameraController()
image_analyzer = ImageAnalyzer()
