"""
Fractal + RSI Strategy for XAUUSD
Main trading strategy implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import MetaTrader5 as mt5
from .base_strategy import BaseStrategy

class FractalRSIStrategy(BaseStrategy):
    """Fractal + RSI strategy implementation matching current EA logic"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.rsi_period = config.get('rsi_period', 14)
        self.fractal_period = config.get('fractal_period', 5)
        self.rsi_up = config.get('rsi_up', 55)
        self.rsi_down = config.get('rsi_down', 45)
        self.symbol = config.get('symbol', 'XAUUSD')
    
    def analyze(self, bars: int = 100) -> Dict:
        """Analyze market data for Fractal + RSI signals (current EA style)"""
        
        # Get market data from MT5 directly (like current implementation)
        timeframe = self._get_timeframe_enum()
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
        
        if rates is None or len(rates) < max(self.rsi_period, self.fractal_period * 2) + 10:
            return {"error": "Insufficient market data", "bars": len(rates) if rates else 0}
        
        # Convert to DataFrame
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        
        # Validate data quality
        if not self._validate_market_data(data):
            return {"error": "Market data validation failed"}
        
        # Calculate indicators (matching current EA logic)
        rsi = self._calculate_rsi(data)
        fractal_up, fractal_down = self._find_fractals(data)
        
        if rsi.empty or fractal_up.empty or fractal_down.empty:
            return {"error": "Indicator calculation failed"}
        
        current_rsi = rsi.iloc[-1]
        
        # Check for recent fractals (within last N bars)
        lookback = self.fractal_period
        latest_fractal_up = fractal_up.iloc[-lookback:].any()
        latest_fractal_down = fractal_down.iloc[-lookback:].any()
        
        signals = {}
        
        # BUY Signal: Fractal Down + RSI > RSI_UP (exact current logic)
        if latest_fractal_down and current_rsi > self.rsi_up:
            signals["BUY"] = {
                "rsi": current_rsi,
                "rsi_threshold": self.rsi_up,
                "fractal_down": True,
                "strength": min(100, (current_rsi - self.rsi_up) * 2),
                "timestamp": datetime.now()
            }
        
        # SELL Signal: Fractal Up + RSI < RSI_DOWN (exact current logic)
        if latest_fractal_up and current_rsi < self.rsi_down:
            signals["SELL"] = {
                "rsi": current_rsi,
                "rsi_threshold": self.rsi_down,
                "fractal_up": True,
                "strength": min(100, (self.rsi_down - current_rsi) * 2),
                "timestamp": datetime.now()
            }
        
        # Add current market info (matching current format)
        signals["market_info"] = {
            "current_rsi": current_rsi,
            "spread": self._get_current_spread(),
            "price": data['close'].iloc[-1],
            "time": data['time'].iloc[-1],
            "bars_analyzed": len(data)
        }
        
        return signals
    
    def validate_signal(self, signal: Dict) -> bool:
        """Validate Fractal + RSI signal (current EA validation)"""
        if 'type' not in signal:
            return False
        
        signal_type = signal['type']
        
        # Validate RSI levels (exact current logic)
        if signal_type == "BUY":
            return signal.get('rsi', 0) > self.rsi_up and signal.get('fractal_down', False)
        elif signal_type == "SELL":
            return signal.get('rsi', 100) < self.rsi_down and signal.get('fractal_up', False)
        
        return False
    
    def _get_timeframe_enum(self) -> int:
        """Get MT5 timeframe enum (from config)"""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        primary_tf = self.config.get('primary_tf', 'M15')
        return tf_map.get(primary_tf, mt5.TIMEFRAME_M15)
    
    def _validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate market data quality (current EA validation)"""
        try:
            # Check required columns
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check for NaN values
            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                return False
            
            # Check price consistency
            if not ((df['high'] >= df['low']) & 
                   (df['high'] >= df['open']) & 
                   (df['high'] >= df['close']) &
                   (df['low'] <= df['open']) & 
                   (df['low'] <= df['close'])).all():
                return False
            
            # Check for zero prices
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator (exact current EA logic)"""
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill NaN values
        rsi = rsi.fillna(50)  # Neutral RSI for missing values
        
        return rsi
    
    def _find_fractals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Find fractal highs and lows (exact current EA logic)"""
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
    
    def _get_current_spread(self) -> float:
        """Get current spread with error handling"""
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            return symbol_info.spread if symbol_info else 30  # Default fallback
        except:
            return 30
