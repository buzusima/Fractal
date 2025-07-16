"""
Fractal + RSI Strategy for XAUUSD
Main trading strategy implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .base_strategy import BaseStrategy

class FractalRSIStrategy(BaseStrategy):
    """Fractal + RSI strategy implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.rsi_period = config.get('rsi_period', 14)
        self.fractal_period = config.get('fractal_period', 5)
        self.rsi_up = config.get('rsi_up', 55)
        self.rsi_down = config.get('rsi_down', 45)
    
    def analyze(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market data for Fractal + RSI signals"""
        
        if len(market_data) < max(self.rsi_period, self.fractal_period * 2) + 10:
            return {"signal": "NO_SIGNAL", "reason": "Insufficient data"}
        
        # Calculate RSI
        rsi = self._calculate_rsi(market_data)
        current_rsi = rsi.iloc[-1]
        
        # Find fractals
        fractal_up, fractal_down = self._find_fractals(market_data)
        
        # Check for recent fractals
        lookback = self.fractal_period
        recent_fractal_up = fractal_up.iloc[-lookback:].any()
        recent_fractal_down = fractal_down.iloc[-lookback:].any()
        
        signals = {}
        
        # BUY Signal: Fractal Down + RSI > RSI_UP
        if recent_fractal_down and current_rsi > self.rsi_up:
            signals["BUY"] = {
                "type": "BUY",
                "rsi": current_rsi,
                "rsi_threshold": self.rsi_up,
                "fractal_down": True,
                "strength": min(100, (current_rsi - self.rsi_up) * 2),
                "timestamp": datetime.now()
            }
        
        # SELL Signal: Fractal Up + RSI < RSI_DOWN
        if recent_fractal_up and current_rsi < self.rsi_down:
            signals["SELL"] = {
                "type": "SELL", 
                "rsi": current_rsi,
                "rsi_threshold": self.rsi_down,
                "fractal_up": True,
                "strength": min(100, (self.rsi_down - current_rsi) * 2),
                "timestamp": datetime.now()
            }
        
        # Add market info
        signals["market_info"] = {
            "current_rsi": current_rsi,
            "price": market_data['close'].iloc[-1],
            "time": market_data['time'].iloc[-1] if 'time' in market_data.columns else datetime.now(),
            "bars_analyzed": len(market_data)
        }
        
        return signals
    
    def validate_signal(self, signal: Dict) -> bool:
        """Validate Fractal + RSI signal"""
        
        if 'type' not in signal:
            return False
        
        signal_type = signal['type']
        
        # Validate RSI levels
        if signal_type == "BUY":
            return signal.get('rsi', 0) > self.rsi_up and signal.get('fractal_down', False)
        elif signal_type == "SELL":
            return signal.get('rsi', 100) < self.rsi_down and signal.get('fractal_up', False)
        
        return False
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator"""
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _find_fractals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Find fractal highs and lows"""
        high = data['high']
        low = data['low']
        
        fractal_up = pd.Series(False, index=data.index)
        fractal_down = pd.Series(False, index=data.index)
        
        for i in range(self.fractal_period, len(data) - self.fractal_period):
            # Fractal Up (Resistance)
            if all(high.iloc[i] >= high.iloc[i-j] for j in range(1, self.fractal_period+1)) and \
               all(high.iloc[i] >= high.iloc[i+j] for j in range(1, self.fractal_period+1)):
                fractal_up.iloc[i] = True
            
            # Fractal Down (Support)
            if all(low.iloc[i] <= low.iloc[i-j] for j in range(1, self.fractal_period+1)) and \
               all(low.iloc[i] <= low.iloc[i+j] for j in range(1, self.fractal_period+1)):
                fractal_down.iloc[i] = True
        
        return fractal_up, fractal_down
