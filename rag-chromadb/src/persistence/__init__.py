"""Persistence layer - data storage"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import os
from src.utils import get_logger

logger = get_logger()


class Storage(ABC):
    """Abstract storage interface"""

    @abstractmethod
    def save(self, data: Any, key: str) -> None:
        """Save data"""
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data"""
        pass


class JSONStorage(Storage):
    """Save/load from JSON files"""

    def __init__(self, data_dir: str = "./json_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"✅ JSON Storage initialized at {data_dir}")

    def save(self, data: Any, key: str) -> None:
        """Save data to JSON file"""
        try:
            path = os.path.join(self.data_dir, f"{key}.json")
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"💾 Saved: {key}")
        except Exception as e:
            logger.error(f"Error saving {key}: {e}")

    def load(self, key: str) -> Optional[Any]:
        """Load data from JSON file"""
        try:
            path = os.path.join(self.data_dir, f"{key}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            return None

    def list_keys(self) -> list:
        """List all stored keys"""
        try:
            files = os.listdir(self.data_dir)
            return [f[:-5] for f in files if f.endswith('.json')]
        except Exception as e:
            logger.error(f"Error listing keys: {e}")
            return []


__all__ = ["Storage", "JSONStorage"]
