# config_manager.py
import json
import os
from pathlib import Path


class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            if not os.path.exists(self.config_file):
                self._create_default_config()

            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "api_keys": {
                "openai": "your-openai-api-key-here",
                "anthropic": "your-anthropic-api-key-here"
            }
        }

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)

        # Set restrictive permissions on the config file
        if os.name != 'nt':  # Unix-like systems
            os.chmod(self.config_file, 0o600)

    def get_api_keys(self):
        """Retrieve API keys from config"""
        return self.config.get("api_keys", {})

    def update_api_key(self, service, key):
        """Update API key for a specific service"""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}

        self.config["api_keys"][service] = key

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)