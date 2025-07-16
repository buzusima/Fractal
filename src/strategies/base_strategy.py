"""
Base Strategy Class for XAUUSD EA
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
        self.last_signal = None
        self.signal_history = []
    
    @abstractmethod
    def analyze(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data and generate signals
        
        Args:
            market_data: OHLCV data
            
        Returns:
            Dict with signal information
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate signal before execution
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if signal is valid
        """
        pass
    
    def get_signal_strength(self, signal: Dict) -> float:
        """Calculate signal strength (0-100)"""
        return signal.get('strength', 50.0)
    
    def log_signal(self, signal: Dict):
        """Log signal to history"""
        signal['timestamp'] = datetime.now()
        signal['strategy'] = self.name
        self.signal_history.append(signal)
        
        # Keep only last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
