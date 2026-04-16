import json
import os

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.default_config = {
            "detection": {
                "conf_thres": 0.35,
                "sim_thres": 0.25,
                "cooldown": 3.0,
                "class_thresholds": {
                    "lie": 0.15,
                    "stand": 0.30,
                    "play_phone": 0.35,
                    "fight": 0.20,
                    "whispering": 0.25,
                    "looking_around": 0.25,
                    "normal": 0.15
                }
            },
            "ui": {
                "window_width": 1300,
                "window_height": 850,
                "theme": "light"
            },
            "paths": {
                "yolo_weight": "yolov8s.pt",
                "prototype_path": "prototypes.pkl"
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"配置文件损坏，使用默认配置: {str(e)}")
                return self.default_config.copy()
        return self.default_config.copy()
    
    def save_config(self):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key, value):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()