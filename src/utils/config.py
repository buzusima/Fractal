"""
Configuration Manager for XAUUSD EA
Handles loading and saving of configuration files
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

@dataclass
class TradingConfig:
    """Enhanced trading configuration"""
    # Entry Settings
    lot_size: float = 0.01
    rsi_up: int = 55
    rsi_down: int = 45
    rsi_period: int = 14
    fractal_period: int = 5
    trading_direction: int = 0  # 0=BOTH, 1=BUY_ONLY, 2=SELL_ONLY, 3=STOP
    
    # Take Profit Settings
    tp_first: int = 200
    exit_speed: int = 1  # 0=FAST, 1=MEDIUM, 2=SLOW
    dynamic_tp: bool = True
    
    # Recovery System
    recovery_price: int = 100
    martingale: float = 2.0
    max_recovery: int = 3
    smart_recovery: bool = True
    
    # Risk Management
    daily_loss_limit: float = 100.0
    max_drawdown: float = 10.0
    max_positions: int = 5
    min_margin_level: float = 200.0
    
    # System Settings
    symbol: str = "XAUUSD"
    primary_tf: str = "M15"
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str):
        """Load configuration from JSON file"""
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()  # Return default config if file doesn't exist

class ConfigManager:
    """Configuration management class"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.current_config = None
    
    def load_config(self, config_name: str = "default") -> TradingConfig:
        """Load configuration by name"""
        config_file = self.config_dir / f"{config_name}.json"
        self.current_config = TradingConfig.load_from_file(config_file)
        return self.current_config
    
    def save_config(self, config: TradingConfig, config_name: str = "default"):
        """Save configuration with name"""
        config_file = self.config_dir / f"{config_name}.json"
        config.save_to_file(config_file)
    
    def list_configs(self) -> List[str]:
        """List available configuration files"""
        return [f.stem for f in self.config_dir.glob("*.json")]
