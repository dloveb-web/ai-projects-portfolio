import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from src.ai.models import (
    DefectPredictionResponse,
    EquipmentHealthResponse,
    RepairDecisionResponse,
    YieldPredictionResponse,
    SchedulingOptimizationResponse
)


class AIService:
    @staticmethod
    def predict_defect(
        product_code: str,
        process_step: str,
        equipment_id: Optional[int] = None,
        operator_id: Optional[int] = None,
        environmental_data: Optional[Dict[str, float]] = None
    ) -> DefectPredictionResponse:
        defect_types = ["mura", "dead_pixel", "line_defect", "color_uniformity", "brightness_issue"]
        probability = random.uniform(0.01, 0.3)
        
        recommendations = []
        if probability > 0.2:
            recommendations.append("建议增加抽检比例")
            recommendations.append("检查设备校准状态")
            recommendations.append("确认操作人员资质")
        elif probability > 0.1:
            recommendations.append("保持常规检查频率")
        else:
            recommendations.append("当前质量风险较低")
        
        return DefectPredictionResponse(
            product_code=product_code,
            defect_probability=round(probability, 4),
            defect_type=defect_types[random.randint(0, len(defect_types)-1)] if probability > 0.15 else None,
            confidence=round(random.uniform(0.75, 0.95), 2),
            recommendations=recommendations
        )

    @staticmethod
    def predict_equipment_health(equipment_id: int, look_ahead_days: int = 7) -> EquipmentHealthResponse:
        equipment_names = {
            1: "大板切割机 #1",
            2: "COG绑定机 #1", 
            3: "FOG绑定机 #1",
            4: "玻璃贴合机 #1",
            5: "背光组装机 #1",
            6: "AOI检测机 #1",
            7: "UV固化炉 #1",
            8: "清洗机 #1"
        }
        
        health_scores = [0.95, 0.88, 0.72, 0.65, 0.92, 0.85, 0.78, 0.91]
        health_score = health_scores[(equipment_id - 1) % len(health_scores)] if equipment_id <= 8 else random.uniform(0.5, 1.0)
        
        if health_score > 0.85:
            risk_level = "低"
            predicted_failure_date = None
            actions = ["设备运行状态良好，继续监控"]
        elif health_score > 0.7:
            risk_level = "中"
            days_to_failure = random.randint(look_ahead_days + 1, look_ahead_days + 14)
            predicted_failure_date = (datetime.now() + timedelta(days=days_to_failure)).strftime("%Y-%m-%d")
            actions = ["建议近期安排预防性维护", "准备备用零件"]
        else:
            risk_level = "高"
            days_to_failure = random.randint(1, look_ahead_days)
            predicted_failure_date = (datetime.now() + timedelta(days=days_to_failure)).strftime("%Y-%m-%d")
            actions = ["立即安排维护", "启动备用设备"]
        
        return EquipmentHealthResponse(
            equipment_id=equipment_id,
            equipment_name=equipment_names.get(equipment_id, f"设备_{equipment_id}"),
            health_score=round(health_score, 4),
            risk_level=risk_level,
            predicted_failure_date=predicted_failure_date,
            recommended_actions=actions
        )

    @staticmethod
    def make_repair_decision(
        product_code: str,
        defect_type: str,
        defect_severity: str,
        repair_history: Optional[List[Dict[str, Any]]] = None
    ) -> RepairDecisionResponse:
        repair_counts = len(repair_history) if repair_history else 0
        
        if defect_severity == "critical":
            should_repair = False
            alternative_action = "建议报废处理"
            confidence = 0.95
            cost_estimate = 0.0
            time_estimate = 0.0
        elif defect_severity == "major":
            if repair_counts >= 2:
                should_repair = False
                alternative_action = "已多次返修，建议报废"
                confidence = 0.85
                cost_estimate = 0.0
                time_estimate = 0.0
            else:
                should_repair = True
                alternative_action = None
                confidence = 0.9
                cost_estimate = random.uniform(50, 200)
                time_estimate = random.uniform(15, 45)
        elif defect_severity == "minor":
            should_repair = True
            alternative_action = None
            confidence = 0.95
            cost_estimate = random.uniform(10, 50)
            time_estimate = random.uniform(5, 15)
        else:
            should_repair = True
            alternative_action = None
            confidence = 0.8
            cost_estimate = random.uniform(20, 100)
            time_estimate = random.uniform(10, 30)
        
        return RepairDecisionResponse(
            product_code=product_code,
            should_repair=should_repair,
            repair_cost_estimate=round(cost_estimate, 2),
            repair_time_estimate=round(time_estimate, 1),
            alternative_action=alternative_action,
            confidence=confidence
        )

    @staticmethod
    def predict_yield(
        product_type: str,
        batch_size: int,
        process_parameters: Optional[Dict[str, float]] = None
    ) -> YieldPredictionResponse:
        base_yield = random.uniform(0.92, 0.98)
        
        key_factors = ["设备稳定性", "操作人员熟练度", "环境温湿度", "物料质量"]
        optimization_suggestions = []
        
        if base_yield < 0.95:
            optimization_suggestions.append("建议优化COG绑定参数")
            optimization_suggestions.append("增加FOG检测频次")
        if random.random() > 0.5:
            optimization_suggestions.append("检查背光模组批次质量")
        
        return YieldPredictionResponse(
            product_type=product_type,
            predicted_yield=round(base_yield, 4),
            confidence=round(random.uniform(0.85, 0.98), 2),
            key_factors=key_factors,
            optimization_suggestions=optimization_suggestions
        )

    @staticmethod
    def optimize_scheduling(
        order_priority: str = "high",
        resource_constraints: Optional[Dict[str, Any]] = None
    ) -> SchedulingOptimizationResponse:
        optimized_schedule = []
        for i in range(5):
            optimized_schedule.append({
                "order_id": f"WO{2024001 + i}",
                "start_time": (datetime.now() + timedelta(hours=i * 4)).strftime("%Y-%m-%d %H:00"),
                "end_time": (datetime.now() + timedelta(hours=(i + 1) * 4 - 1)).strftime("%Y-%m-%d %H:00"),
                "workstation": f"WS-{i % 3 + 1}",
                "priority": order_priority
                       })
        
        bottleneck_analysis = []
        if random.random() > 0.6:
            bottleneck_analysis.append("COG工位负载较高，建议分流")
        if random.random() > 0.7:
            bottleneck_analysis.append("UV固化炉等待时间较长")
        
        return SchedulingOptimizationResponse(
            optimized_schedule=optimized_schedule,
            expected_throughput=round(random.uniform(2400, 3200), 0),
            bottleneck_analysis=bottleneck_analysis
        )

    @staticmethod
    def analyze_defect_trends(hours: int = 24) -> Dict[str, Any]:
        defect_types = ["mura", "dead_pixel", "line_defect", "color_uniformity", "brightness_issue"]
        defect_type_names = {
            "mura": "Mura缺陷",
            "dead_pixel": "坏点", 
            "line_defect": "线缺陷",
            "color_uniformity": "色不均",
            "brightness_issue": "亮度问题"
        }
        
        base_defect_rates = [0.025, 0.018, 0.015, 0.012, 0.008]
        trend_data = []
        
        for i in range(min(hours, 24)):
            hour = (datetime.now() - timedelta(hours=i)).strftime("%H:00")
            base_rate = base_defect_rates[i % len(base_defect_rates)]
            variation = random.uniform(-0.005, 0.008)
            
            trend_data.append({
                "hour": hour,
                "defect_count": int((base_rate + variation) * 1000),
                "defect_rate": round(base_rate + variation, 4)
            })
        
        trend_data.reverse()
        
        top_defects = [
            {"type": defect_type_names["mura"], "count": 45, "rate": 0.028},
            {"type": defect_type_names["dead_pixel"], "count": 32, "rate": 0.020},
            {"type": defect_type_names["line_defect"], "count": 25, "rate": 0.016}
        ]
        
        return {
            "trend_data": trend_data,
            "top_defects": top_defects,
            "overall_defect_rate": round(sum(d["defect_rate"] for d in trend_data) / len(trend_data), 4),
            "anomaly_detection": [
                {"time": "14:00", "type": "spike", "value": "缺陷率异常上升"},
                {"time": "18:00", "type": "warning", "value": "设备维护提醒"}
            ]
        }

    @staticmethod
    def generate_quality_report(product_type: str = "all") -> Dict[str, Any]:
        return {
            "report_period": "今日",
            "total_units": 10850,
            "good_units": 10558,
            "defect_units": 292,
            "yield_rate": 0.973,
            "defect_distribution": [
                {"type": "Mura缺陷", "count": 125, "percentage": 42.8},
                {"type": "坏点", "count": 85, "percentage": 29.1},
                {"type": "线缺陷", "count": 52, "percentage": 17.8},
                {"type": "色不均", "count": 20, "percentage": 6.8},
                {"type": "其他", "count": 10, "percentage": 3.5}
            ],
            "recommendations": [
                "关注Mura缺陷上升趋势，建议检查偏光片贴合工艺",
                "坏点缺陷率维持稳定，继续保持当前管控水平",
                "建议增加COG工位的AOI检测频率至每小时一次",
                "背光模组批次质量良好，可继续使用当前供应商"
            ],
            "ai_summary": "AI分析显示当前综合良率97.3%处于正常水平，但Mura缺陷在近4小时有上升趋势，建议重点关注前段工艺，特别是大板切割后的清洗工序。COG绑定设备的健康度略降，建议安排预防性维护。"
        }
