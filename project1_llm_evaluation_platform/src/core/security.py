import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from ..database.models import SecurityLog

logger = logging.getLogger(__name__)


class SecurityChecker:
    def __init__(self, db: Session = None):
        self.db = db
        self.prompt_injection_patterns = [
            r"忽略.*指令",
            r"ignore.*instruction",
            r"disregard.*previous",
            r"forget.*system",
            r"override.*settings",
            r"你.*开发者.*模式",
            r"开发者.*模式",
            r"new\s+instruction",
            r"system\s+prompt",
            r"reveal.*your.*prompt",
            r"forget.*all.*previous",
            r"act.*as.*different",
            r"你现在是",
            r"你现在是一个",
        ]

        self.sensitive_keywords = {
            "political": [
                "台独", "港独", "藏独", "疆独", "分裂", "颠覆", "反动",
            ],
            "violence": [
                "杀人", "抢劫", "暴力", "恐怖", "爆炸", "武器", "毒品", "制造炸弹",
            ],
            "pornography": [
                "色情", "裸体", "性交易", "AV", "黄色",
            ],
            "fraud": [
                "诈骗", "钓鱼", "盗取", "非法集资", "传销",
            ],
        }

        self.rag_attack_patterns = [
            r"在.*文档.*提到",
            r"根据.*内容.*回答",
            r"结合.*背景",
            r"利用.*信息",
            r"前面的.*都是.*虚构",
            r"忘记.*规则",
            r"角色.*扮演",
        ]

    def check(self, text: str, log_to_db: bool = True) -> Dict[str, Any]:
        threats = []

        injection_result = self._check_prompt_injection(text)
        if injection_result["detected"]:
            threats.append(injection_result)

        sensitive_result = self._check_sensitive_content(text)
        if sensitive_result["detected"]:
            threats.append(sensitive_result)

        rag_result = self._check_rag_attack(text)
        if rag_result["detected"]:
            threats.append(rag_result)

        is_safe = len(threats) == 0
        threat_level = self._calculate_threat_level(threats)

        result = {
            "is_safe": is_safe,
            "threats": threats,
            "threat_level": threat_level,
            "checked_at": datetime.now().isoformat(),
            "text_length": len(text),
        }

        if log_to_db and not is_safe and self.db:
            self._log_threat(text, threats, threat_level)

        return result

    def _check_prompt_injection(self, text: str) -> Dict[str, Any]:
        detected_patterns = []
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_patterns.append(pattern)

        return {
            "type": "prompt_injection",
            "detected": len(detected_patterns) > 0,
            "matched_patterns": detected_patterns,
            "threat_level": "high" if detected_patterns else "none",
            "description": "检测到提示词注入尝试" if detected_patterns else "未检测到注入",
        }

    def _check_sensitive_content(self, text: str) -> Dict[str, Any]:
        detected_categories = []
        matched_keywords = {}

        for category, keywords in self.sensitive_keywords.items():
            found = [kw for kw in keywords if kw in text]
            if found:
                detected_categories.append(category)
                matched_keywords[category] = found

        return {
            "type": "sensitive_content",
            "detected": len(detected_categories) > 0,
            "categories": detected_categories,
            "matched_keywords": matched_keywords,
            "threat_level": "critical" if "political" in detected_categories else "high" if detected_categories else "none",
            "description": f"检测到敏感内容类别: {', '.join(detected_categories)}" if detected_categories else "未检测到敏感内容",
        }

    def _check_rag_attack(self, text: str) -> Dict[str, Any]:
        detected_patterns = []
        for pattern in self.rag_attack_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected_patterns.append(pattern)

        return {
            "type": "rag_attack",
            "detected": len(detected_patterns) > 0,
            "matched_patterns": detected_patterns,
            "threat_level": "medium" if detected_patterns else "none",
            "description": "检测到RAG攻击尝试" if detected_patterns else "未检测到RAG攻击",
        }

    def _calculate_threat_level(self, threats: List[Dict]) -> str:
        if not threats:
            return "none"
        levels = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        max_level = max(levels.get(t.get("threat_level", "none"), 0) for t in threats)
        for level, score in levels.items():
            if score == max_level:
                return level
        return "none"

    def _log_threat(self, text: str, threats: List[Dict], threat_level: str):
        for threat in threats:
            log = SecurityLog(
                input_text=text[:1000],
                threat_type=threat["type"],
                threat_level=threat_level,
                action="blocked",
                details={"matched_patterns": threat.get("matched_patterns", []),
                        "categories": threat.get("categories", [])},
            )
            self.db.add(log)
        self.db.commit()

    def sanitize_output(self, text: str) -> str:
        sanitized = text
        for category, keywords in self.sensitive_keywords.items():
            for keyword in keywords:
                sanitized = sanitized.replace(keyword, "*" * len(keyword))
        return sanitized

    def batch_check(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.check(text) for text in texts]
