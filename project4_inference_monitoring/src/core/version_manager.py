import hashlib
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VersionManager:
    def __init__(self):
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.active_version: Optional[str] = None
        self.grayscale_config = {
            "enabled": False,
            "percentage": 10,
            "strategy": "random"
        }
        self._initialize_default_versions()

    def _initialize_default_versions(self):
        default_version = {
            "version": "v1.0",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "model_path": "/models/Qwen2.5-7B-Instruct",
            "status": "active",
            "created_at": datetime.now(),
            "traffic_percentage": 100,
            "description": "Default production version"
        }
        self.versions["v1.0"] = default_version
        self.active_version = "v1.0"

    def register_version(
        self,
        version: str,
        model_name: str,
        model_path: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        if version in self.versions:
            raise ValueError(f"Version {version} already exists")

        version_info = {
            "version": version,
            "model_name": model_name,
            "model_path": model_path,
            "status": "pending",
            "created_at": datetime.now(),
            "traffic_percentage": 0,
            "description": description or f"Version {version}"
        }

        self.versions[version] = version_info
        logger.info(f"Registered new version: {version}")
        return version_info

    def activate_version(self, version: str) -> bool:
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        if self.active_version and self.active_version in self.versions:
            self.versions[self.active_version]["status"] = "deprecated"
            self.versions[self.active_version]["traffic_percentage"] = 0

        self.versions[version]["status"] = "active"
        self.versions[version]["traffic_percentage"] = 100
        self.active_version = version

        logger.info(f"Activated version: {version}")
        return True

    def _get_grayscale_version(self) -> Optional[str]:
        for version, info in self.versions.items():
            if info["status"] == "grayscale":
                return version
        return None

    def enable_grayscale(
        self,
        percentage: int = 10,
        strategy: str = "random"
    ) -> Dict[str, Any]:
        if not 0 < percentage <= 100:
            raise ValueError("Percentage must be between 1 and 100")

        self.grayscale_config = {
            "enabled": True,
            "percentage": percentage,
            "strategy": strategy
        }

        if self.active_version and len(self.versions) > 1:
            active = self.active_version
            candidates = [v for v in self.versions.keys() if v != active]

            if candidates:
                new_version = candidates[0]
                self.versions[active]["traffic_percentage"] = 100 - percentage
                self.versions[new_version]["status"] = "grayscale"
                self.versions[new_version]["traffic_percentage"] = percentage

        logger.info(f"Enabled grayscale with {percentage}% traffic to new version")
        return self.grayscale_config

    def disable_grayscale(self) -> bool:
        self.grayscale_config["enabled"] = False

        if self.active_version:
            for v in self.versions.values():
                if v["status"] == "grayscale":
                    v["status"] = "pending"
                    v["traffic_percentage"] = 0

        logger.info("Disabled grayscale deployment")
        return True

    def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        return self.versions.get(version)

    def get_active_version(self) -> Optional[Dict[str, Any]]:
        if self.active_version:
            return self.versions.get(self.active_version)
        return None

    def get_all_versions(self) -> List[Dict[str, Any]]:
        return list(self.versions.values())

    def select_version_for_request(self, user_id: Optional[str] = None) -> str:
        if not self.grayscale_config["enabled"]:
            return self.active_version or "v1.0"

        grayscale_version = self._get_grayscale_version()
        if not grayscale_version:
            return self.active_version or "v1.0"

        percentage = self.grayscale_config["percentage"]
        strategy = self.grayscale_config["strategy"]

        if strategy == "random":
            import random
            return grayscale_version if random.randint(1, 100) <= percentage else self.active_version

        elif strategy == "user_hash":
            if not user_id:
                return self.active_version
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
            return grayscale_version if hash_value < percentage else self.active_version

        else:
            return self.active_version

    def rollback_version(self) -> bool:
        candidates = [
            (v, info) for v, info in self.versions.items()
            if v != self.active_version and info["status"] in ["deprecated", "grayscale"]
        ]

        if not candidates:
            logger.warning("No version available for rollback")
            return False

        last_version, _ = candidates[-1]
        return self.activate_version(last_version)

    def get_traffic_distribution(self) -> Dict[str, int]:
        return {
            version: info["traffic_percentage"]
            for version, info in self.versions.items()
        }


_global_version_manager = VersionManager()


def get_version_manager() -> VersionManager:
    return _global_version_manager
