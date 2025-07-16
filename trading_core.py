import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class TradingConfig:
    """Configuration class for UI-adjustable parameters"""
    # Entry Settings
    lot_size: float = 0.01
    rsi_up: int = 55
    rsi_down: int = 45
    rsi_period: int = 14
    fractal_period: int = 5
    trading_direction: int = 0  # 0=BOTH, 1=BUY_ONLY, 2=SELL_ONLY, 3=STOP
    
    # Take Profit Settings
    tp_first: int = 200  # Points for first position
    exit_speed: int = 1  # 0=FAST, 1=MEDIUM, 2=SLOW
    dynamic_tp: bool = True
    
    # Recovery System
    recovery_price: int = 100  # Points loss to trigger recovery
    martingale: float = 2.0
    max_recovery: int = 3
    smart_recovery: bool = True
    
    # Spread Management
    spread_mode: int = 0  # 0=AUTO, 1=FIXED, 2=SMART, 3=NONE
    spread_buffer: int = 5
    max_spread_alert: int = 30
    
    # Timeframe Settings
    primary_tf: str = "M15"
    tf_mode: int = 0  # 0=SINGLE, 1=MULTI, 2=CASCADE, 3=ADAPTIVE
    
    # Risk Management
    daily_loss_limit: float = 100.0
    max_positions: int = 5
    max_drawdown: float = 10.0
    
    # System Settings
    symbol: str = "XAUUSD.v"
    auto_symbol_detect: bool = True
    
    def update_from_dict(self, params: Dict):
        """Update parameters from dictionary (for UI updates)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for UI display"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def get_timeframe_enum(self) -> int:
        """Convert timeframe string to MT5 enum"""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        return tf_map.get(self.primary_tf, mt5.TIMEFRAME_M15)

@dataclass
class AccountInfo:
    """Account information structure"""
    login: int = 0
    server: str = ""
    name: str = ""
    company: str = ""
    currency: str = "USD"
    leverage: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    trade_allowed: bool = False
    is_demo: bool = True

@dataclass 
class SymbolInfo:
    """Symbol information structure"""
    name: str = ""
    description: str = ""
    currency_base: str = ""
    currency_profit: str = ""
    currency_margin: str = ""
    digits: int = 0
    point: float = 0.0
    spread: int = 0
    volume_min: float = 0.0
    volume_max: float = 0.0
    volume_step: float = 0.0
    contract_size: float = 0.0
    margin_initial: float = 0.0
    margin_maintenance: float = 0.0
    trade_mode: int = 0
    filling_mode: int = 0
    expiration_mode: int = 0

class XAUUSDTradingCore:
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        
        # Account and symbol validation
        self.account_info = AccountInfo()
        self.symbol_info = SymbolInfo()
        self.is_connected = False
        self.is_validated = False
        
        # Internal tracking
        self.positions = {}
        self.recovery_levels = {}
        self.last_signals = {}
        self.is_trading = False
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Point calculation for different brokers
        self.point_multiplier = 1.0  # Will be calculated based on broker
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_mt5(self) -> bool:
        """Enhanced MT5 initialization with full validation"""
        self.logger.info("Initializing MT5 connection...")
        
        # Step 1: Basic MT5 initialization
        if not mt5.initialize():
            error_msg = f"MT5 initialization failed: {mt5.last_error()}"
            self.logger.error(error_msg)
            return False
        
        self.logger.info("✓ MT5 basic initialization successful")
        
        # Step 2: Get and validate account info
        if not self._validate_account():
            self.logger.error("Account validation failed")
            mt5.shutdown()
            return False
        
        # Step 3: Auto-detect and validate symbol
        if not self._detect_and_validate_symbol():
            self.logger.error("Symbol detection/validation failed")
            mt5.shutdown()
            return False
        
        # Step 4: Calculate point values for this broker
        self._calculate_point_values()
        
        # Step 5: Final connection validation
        self.is_connected = True
        self.is_validated = True
        
        self.logger.info("🎉 MT5 initialization completed successfully")
        self._log_connection_summary()
        
        return True
    
    def _validate_account(self) -> bool:
        """Validate account information with proper MT5 attribute handling"""
        try:
            account = mt5.account_info()
            if account is None:
                self.logger.error("Cannot get account information")
                return False
            
            # Safe attribute access with getattr and defaults
            login = getattr(account, 'login', 0)
            server = getattr(account, 'server', '')
            name = getattr(account, 'name', '')
            company = getattr(account, 'company', '')
            currency = getattr(account, 'currency', 'USD')
            leverage = getattr(account, 'leverage', 0)
            balance = getattr(account, 'balance', 0.0)
            equity = getattr(account, 'equity', 0.0)
            margin = getattr(account, 'margin', 0.0)
            
            # Handle free_margin - some MT5 versions don't have this attribute
            free_margin = getattr(account, 'free_margin', None)
            if free_margin is None:
                # Calculate free margin if not available
                free_margin = equity - margin if equity > margin else equity
            
            # Calculate margin level safely
            margin_level = 999.99  # Default for no positions
            if margin > 0:
                margin_level = (equity / margin) * 100
            
            # Check trade_allowed attribute safely
            trade_allowed = getattr(account, 'trade_allowed', True)
            
            # Check account trade mode safely
            trade_mode = getattr(account, 'trade_mode', mt5.ACCOUNT_TRADE_MODE_DEMO)
            is_demo = (trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO)
            
            # Populate account info
            self.account_info = AccountInfo(
                login=login,
                server=server,
                name=name,
                company=company,
                currency=currency,
                leverage=leverage,
                balance=float(balance),
                equity=float(equity),
                margin=float(margin),
                free_margin=float(free_margin),
                margin_level=float(margin_level),
                trade_allowed=trade_allowed,
                is_demo=is_demo
            )
            
            # Validation checks
            if not self.account_info.trade_allowed:
                self.logger.error("❌ Trading not allowed on this account")
                return False
            
            if self.account_info.balance <= 0:
                self.logger.error("❌ Account balance is zero or negative")
                return False
            
            if self.account_info.leverage <= 0:
                self.logger.warning("⚠️ Leverage information not available")
            
            # Check margin level
            if self.account_info.margin > 0 and self.account_info.margin_level < 100:
                self.logger.warning(f"⚠️ Low margin level: {self.account_info.margin_level:.2f}%")
            
            self.logger.info(f"✓ Account validated: {self.account_info.name} ({self.account_info.server})")
            self.logger.info(f"  Balance: ${self.account_info.balance:.2f} {self.account_info.currency}")
            self.logger.info(f"  Equity: ${self.account_info.equity:.2f}")
            self.logger.info(f"  Free Margin: ${self.account_info.free_margin:.2f}")
            self.logger.info(f"  Margin Level: {self.account_info.margin_level:.2f}%")
            self.logger.info(f"  Leverage: 1:{self.account_info.leverage}")
            self.logger.info(f"  Account Type: {'DEMO' if self.account_info.is_demo else 'LIVE'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Account validation error: {e}")
            self.logger.error(f"Available account attributes: {dir(account) if 'account' in locals() else 'No account object'}")
            return False
        
    def _detect_and_validate_symbol(self) -> bool:
        """Auto-detect and validate XAUUSD symbol with proper MT5 constants"""
        try:
            # Auto-detect symbol if enabled
            if self.config.auto_symbol_detect:
                detected_symbol = self._detect_gold_symbol()
                if detected_symbol:
                    self.config.symbol = detected_symbol
                    self.logger.info(f"✓ Auto-detected symbol: {self.config.symbol}")
            
            # Get symbol information with error handling
            symbol = mt5.symbol_info(self.config.symbol)
            if symbol is None:
                self.logger.error(f"❌ Symbol {self.config.symbol} not found")
                return False
            
            # Enable symbol if not visible
            if not symbol.visible:
                if not mt5.symbol_select(self.config.symbol, True):
                    self.logger.error(f"❌ Failed to select symbol {self.config.symbol}")
                    return False
                self.logger.info(f"✓ Symbol {self.config.symbol} enabled")
            
            # Get filling mode safely - use actual MT5 constants
            filling_mode = getattr(symbol, 'filling_mode', 0)
            # Check for valid filling modes that exist in MT5
            if hasattr(mt5, 'SYMBOL_FILLING_FOK') and (filling_mode & mt5.SYMBOL_FILLING_FOK):
                default_filling = mt5.SYMBOL_FILLING_FOK
            elif hasattr(mt5, 'SYMBOL_FILLING_IOC') and (filling_mode & mt5.SYMBOL_FILLING_IOC):
                default_filling = mt5.SYMBOL_FILLING_IOC
            else:
                # Use numeric value as fallback
                default_filling = 1  # IOC equivalent
            
            # Get trade mode safely
            trade_mode = getattr(symbol, 'trade_mode', 4)  # 4 = SYMBOL_TRADE_MODE_FULL
            
            # Populate symbol info with safe attribute access
            self.symbol_info = SymbolInfo(
                name=getattr(symbol, 'name', self.config.symbol),
                description=getattr(symbol, 'description', ''),
                currency_base=getattr(symbol, 'currency_base', 'XAU'),
                currency_profit=getattr(symbol, 'currency_profit', 'USD'),
                currency_margin=getattr(symbol, 'currency_margin', 'USD'),
                digits=getattr(symbol, 'digits', 2),
                point=getattr(symbol, 'point', 0.01),
                spread=getattr(symbol, 'spread', 30),
                volume_min=getattr(symbol, 'volume_min', 0.01),
                volume_max=getattr(symbol, 'volume_max', 100.0),
                volume_step=getattr(symbol, 'volume_step', 0.01),
                contract_size=getattr(symbol, 'contract_size', 100.0),
                margin_initial=getattr(symbol, 'margin_initial', 0.0),
                margin_maintenance=getattr(symbol, 'margin_maintenance', 0.0),
                trade_mode=trade_mode,
                filling_mode=default_filling,
                expiration_mode=getattr(symbol, 'expiration_mode', 1)  # GTC equivalent
            )
            
            # Validation checks - use numeric values to avoid constant issues
            if self.symbol_info.trade_mode == 0:  # SYMBOL_TRADE_MODE_DISABLED
                self.logger.error("❌ Trading disabled for this symbol")
                return False
            
            if self.symbol_info.volume_min <= 0:
                self.logger.error("❌ Invalid minimum volume")
                return False
            
            # Check if it's really gold (optional warning, not blocking)
            if not self._is_gold_symbol():
                self.logger.warning(f"⚠️ Symbol {self.config.symbol} may not be gold")
            
            self.logger.info(f"✓ Symbol validated: {self.symbol_info.description}")
            self.logger.info(f"  Digits: {self.symbol_info.digits}, Point: {self.symbol_info.point}")
            self.logger.info(f"  Volume range: {self.symbol_info.volume_min} - {self.symbol_info.volume_max}")
            self.logger.info(f"  Current spread: {self.symbol_info.spread} points")
            self.logger.info(f"  Contract size: {self.symbol_info.contract_size}")
            self.logger.info(f"  Trade mode: {self.symbol_info.trade_mode}")
            self.logger.info(f"  Filling mode: {self.symbol_info.filling_mode}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Symbol validation error: {e}")
            return False

    def _is_gold_symbol(self) -> bool:
        """Check if current symbol is gold - simplified version"""
        try:
            symbol_name = self.config.symbol.upper()
            
            # Check for gold indicators in symbol name
            gold_indicators = ['XAU', 'GOLD']
            return any(indicator in symbol_name for indicator in gold_indicators)
            
        except Exception as e:
            self.logger.error(f"Gold symbol check error: {e}")
            return True  # Assume it's gold if check fails

    def _validate_gold_symbol(self, symbol_info) -> bool:
        """Validate if symbol is actually gold - simplified version"""
        try:
            # Check symbol name contains gold indicators
            name_lower = getattr(symbol_info, 'name', '').lower()
            if not any(indicator in name_lower for indicator in ['xau', 'gold']):
                return False
            
            # Basic validation - just check if it looks like gold
            digits = getattr(symbol_info, 'digits', 0)
            if digits > 0 and digits <= 5:
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Gold validation error: {e}")
            return True  # Assume valid if validation fails
    
    def _detect_gold_symbol(self) -> Optional[str]:
        """Auto-detect gold symbol variations with better error handling"""
        gold_symbols = [
            "XAUUSD", "XAUUSD.m", "XAUUSD.raw", "XAUUSD.v", "XAUUSD.c",
            "#XAUUSD", "GOLD", "GOLDUSD", "XAU/USD", "XAUUSD.",
            "Gold", "GOLD.m", "XAUUSD.a", "XAUUSD.b", "XAUUSD.mt5"
        ]
        
        self.logger.info("Auto-detecting gold symbol...")
        
        for symbol in gold_symbols:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    # Additional validation - check if it's really gold
                    if self._validate_gold_symbol(symbol_info):
                        self.logger.info(f"✓ Gold symbol found: {symbol}")
                        return symbol
                    else:
                        self.logger.debug(f"Symbol {symbol} found but doesn't match gold characteristics")
            except Exception as e:
                self.logger.debug(f"Error checking symbol {symbol}: {e}")
                continue
        
        self.logger.warning("❌ No gold symbol auto-detected, using default")
        return None

    def _calculate_point_values(self):
        """Calculate point values and multipliers for this broker with error handling"""
        try:
            # Get current tick
            tick = mt5.symbol_info_tick(self.config.symbol)
            if tick is None:
                self.logger.warning("Cannot get current tick for point calculation")
                # Use symbol info point value as fallback
                self.point_multiplier = 1.0
                return
            
            # Calculate point value based on contract size
            contract_size = getattr(self.symbol_info, 'contract_size', 100.0)
            point_value = getattr(self.symbol_info, 'point', 0.01)
            
            # Standard XAUUSD: 1 lot = 100 oz, 1 point = $1 per lot
            point_value_per_lot = contract_size * point_value
            
            # For XAUUSD, point value should be around $0.01 per 0.01 lot
            expected_point_value = 0.01  # $0.01 per 0.01 lot per point
            actual_point_value = point_value_per_lot / 100  # per 0.01 lot
            
            if expected_point_value > 0:
                self.point_multiplier = actual_point_value / expected_point_value
            else:
                self.point_multiplier = 1.0
            
            self.logger.info(f"✓ Point calculation completed:")
            self.logger.info(f"  Contract size: {contract_size}")
            self.logger.info(f"  Point value: {point_value}")
            self.logger.info(f"  Point value per lot: ${point_value_per_lot}")
            self.logger.info(f"  Point value per 0.01 lot: ${actual_point_value:.4f}")
            self.logger.info(f"  Point multiplier: {self.point_multiplier:.4f}")
            
            # Warn if unusual values
            if not (0.1 <= self.point_multiplier <= 10.0):
                self.logger.warning(f"⚠️ Unusual point multiplier: {self.point_multiplier}")
            
        except Exception as e:
            self.logger.error(f"Point calculation error: {e}")
            self.point_multiplier = 1.0  # Default fallback

    def _detect_gold_symbol(self) -> Optional[str]:
        """Auto-detect gold symbol variations"""
        gold_symbols = [
            "XAUUSD", "XAUUSD.m", "XAUUSD.raw", "XAUUSD.v", "XAUUSD.c",
            "#XAUUSD", "GOLD", "GOLDUSD", "XAU/USD", "XAUUSD.",
            "Gold", "GOLD.m", "XAUUSD.a", "XAUUSD.b"
        ]
        
        self.logger.info("Auto-detecting gold symbol...")
        
        for symbol in gold_symbols:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    # Additional validation - check if it's really gold
                    if self._validate_gold_symbol(symbol_info):
                        self.logger.info(f"✓ Gold symbol found: {symbol}")
                        return symbol
                    else:
                        self.logger.debug(f"Symbol {symbol} found but doesn't match gold characteristics")
            except:
                continue
        
        self.logger.warning("❌ No gold symbol auto-detected, using default")
        return None
    
    def _validate_gold_symbol(self, symbol_info) -> bool:
        """Validate if symbol is actually gold"""
        try:
            # Check symbol name contains gold indicators
            name_lower = symbol_info.name.lower()
            if not any(indicator in name_lower for indicator in ['xau', 'gold']):
                return False
            
            # Check base currency (should be XAU or similar)
            if hasattr(symbol_info, 'currency_base'):
                base_currency = symbol_info.currency_base.upper()
                if base_currency not in ['XAU', 'GOLD']:
                    return False
            
            # Check profit currency (should be USD for XAUUSD)
            if hasattr(symbol_info, 'currency_profit'):
                profit_currency = symbol_info.currency_profit.upper()
                if profit_currency != 'USD':
                    return False
            
            # Check digits (gold usually has 2-3 digits)
            if not (1 <= symbol_info.digits <= 5):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Gold validation error: {e}")
            return False
    
    def _is_gold_symbol(self) -> bool:
        """Check if current symbol is gold"""
        return self._validate_gold_symbol(mt5.symbol_info(self.config.symbol))
    
    def _calculate_point_values(self):
        """Calculate point values and multipliers for this broker"""
        try:
            # Get current tick
            tick = mt5.symbol_info_tick(self.config.symbol)
            if tick is None:
                self.logger.warning("Cannot get current tick for point calculation")
                return
            
            # Calculate point value based on contract size
            # Standard XAUUSD: 1 lot = 100 oz, 1 point = $1 per lot
            point_value_per_lot = self.symbol_info.contract_size * self.symbol_info.point
            
            # For XAUUSD, point value should be around $0.01 per 0.01 lot
            expected_point_value = 0.01  # $0.01 per 0.01 lot per point
            actual_point_value = point_value_per_lot / 100  # per 0.01 lot
            
            self.point_multiplier = actual_point_value / expected_point_value
            
            self.logger.info(f"✓ Point calculation completed:")
            self.logger.info(f"  Contract size: {self.symbol_info.contract_size}")
            self.logger.info(f"  Point value per lot: ${point_value_per_lot}")
            self.logger.info(f"  Point value per 0.01 lot: ${actual_point_value:.4f}")
            self.logger.info(f"  Point multiplier: {self.point_multiplier:.4f}")
            
            # Warn if unusual values
            if not (0.5 <= self.point_multiplier <= 2.0):
                self.logger.warning(f"⚠️ Unusual point multiplier: {self.point_multiplier}")
            
        except Exception as e:
            self.logger.error(f"Point calculation error: {e}")
            self.point_multiplier = 1.0  # Default fallback
    
    def _log_connection_summary(self):
        """Log connection summary"""
        self.logger.info("=" * 60)
        self.logger.info("MT5 CONNECTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Account: {self.account_info.name} ({self.account_info.login})")
        self.logger.info(f"Server: {self.account_info.server}")
        self.logger.info(f"Company: {self.account_info.company}")
        self.logger.info(f"Type: {'DEMO' if self.account_info.is_demo else 'LIVE'}")
        self.logger.info(f"Currency: {self.account_info.currency}")
        self.logger.info(f"Balance: ${self.account_info.balance:.2f}")
        self.logger.info(f"Leverage: 1:{self.account_info.leverage}")
        self.logger.info("-" * 60)
        self.logger.info(f"Symbol: {self.symbol_info.name}")
        self.logger.info(f"Description: {self.symbol_info.description}")
        self.logger.info(f"Digits: {self.symbol_info.digits}")
        self.logger.info(f"Spread: {self.symbol_info.spread} points")
        self.logger.info(f"Min Volume: {self.symbol_info.volume_min}")
        self.logger.info(f"Volume Step: {self.symbol_info.volume_step}")
        self.logger.info("=" * 60)
    
    def update_config(self, new_params: Dict):
        """Update trading configuration from UI"""
        old_config = self.config.to_dict()
        self.config.update_from_dict(new_params)
        
        # Log changes
        changes = {k: v for k, v in new_params.items() if old_config.get(k) != v}
        if changes:
            self.logger.info(f"Config updated: {changes}")
        
        # Re-validate critical parameters
        if 'symbol' in changes and self.is_connected:
            self.logger.info("Symbol changed, re-validating...")
            self._detect_and_validate_symbol()
    
    def get_config(self) -> Dict:
        """Get current configuration for UI"""
        return self.config.to_dict()
    
    def get_connection_status(self) -> Dict:
        """Get detailed connection status"""
        return {
            "connected": self.is_connected,
            "validated": self.is_validated,
            "account": {
                "login": self.account_info.login,
                "server": self.account_info.server,
                "name": self.account_info.name,
                "balance": self.account_info.balance,
                "equity": self.account_info.equity,
                "margin_level": self.account_info.margin_level,
                "is_demo": self.account_info.is_demo,
                "trade_allowed": self.account_info.trade_allowed
            },
            "symbol": {
                "name": self.symbol_info.name,
                "description": self.symbol_info.description,
                "spread": self.symbol_info.spread,
                "point": self.symbol_info.point,
                "digits": self.symbol_info.digits,
                "min_volume": self.symbol_info.volume_min,
                "max_volume": self.symbol_info.volume_max
            },
            "point_multiplier": self.point_multiplier
        }
    
    def get_market_data(self, bars: int = 100) -> pd.DataFrame:
        """Get market data for analysis with validation"""
        if not self.is_connected:
            self.logger.error("MT5 not connected")
            return None
        
        try:
            timeframe = self.config.get_timeframe_enum()
            rates = mt5.copy_rates_from_pos(self.config.symbol, timeframe, 0, bars)
            
            if rates is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get market data: {error}")
                return None
            
            if len(rates) < bars * 0.8:  # Should get at least 80% of requested bars
                self.logger.warning(f"Got {len(rates)} bars, requested {bars}")
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Validate data quality
            if not self._validate_market_data(df):
                self.logger.error("Market data validation failed")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market data error: {e}")
            return None
    
    def _validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate market data quality"""
        try:
            # Check required columns
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check for NaN values
            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                return False
            
            # Check price consistency (high >= low, etc.)
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
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate RSI indicator with validation"""
        period = period or self.config.rsi_period
        
        if data is None or len(data) < period + 10:
            self.logger.warning(f"Insufficient data for RSI calculation: {len(data) if data is not None else 0}")
            return pd.Series()
        
        try:
            close = data['close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values
            rsi = rsi.fillna(50)  # Neutral RSI for missing values
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return pd.Series()
    
    def find_fractals(self, data: pd.DataFrame, period: int = None) -> Tuple[pd.Series, pd.Series]:
        """Find fractal highs and lows with validation"""
        period = period or self.config.fractal_period
        
        if data is None or len(data) < period * 2 + 10:
            self.logger.warning(f"Insufficient data for fractal calculation: {len(data) if data is not None else 0}")
            return pd.Series(dtype=bool), pd.Series(dtype=bool)
        
        try:
            high = data['high']
            low = data['low']
            
            fractal_up = pd.Series(False, index=data.index)
            fractal_down = pd.Series(False, index=data.index)
            
            for i in range(period, len(data) - period):
                # Fractal Up (Resistance)
                if all(high.iloc[i] >= high.iloc[i-j] for j in range(1, period+1)) and \
                   all(high.iloc[i] >= high.iloc[i+j] for j in range(1, period+1)):
                    fractal_up.iloc[i] = True
                
                # Fractal Down (Support)
                if all(low.iloc[i] <= low.iloc[i-j] for j in range(1, period+1)) and \
                   all(low.iloc[i] <= low.iloc[i+j] for j in range(1, period+1)):
                    fractal_down.iloc[i] = True
            
            return fractal_up, fractal_down
            
        except Exception as e:
            self.logger.error(f"Fractal calculation error: {e}")
            return pd.Series(dtype=bool), pd.Series(dtype=bool)
    
    def get_current_spread(self) -> float:
        """Get current spread in points with validation"""
        if not self.is_connected:
            return 0
        
        try:
            symbol_info = mt5.symbol_info(self.config.symbol)
            if symbol_info is None:
                return 0
            
            spread = symbol_info.spread
            
            # Validate spread
            if spread < 0 or spread > 1000:  # Sanity check
                self.logger.warning(f"Unusual spread value: {spread}")
                
            return spread
            
        except Exception as e:
            self.logger.error(f"Spread error: {e}")
            return 0
    
    def calculate_spread_buffer(self) -> int:
        """Calculate spread buffer based on mode with validation"""
        try:
            current_spread = self.get_current_spread()
            
            if self.config.spread_mode == 0:  # AUTO
                buffer = int(current_spread * 1.5) + 2
            elif self.config.spread_mode == 1:  # FIXED
                buffer = self.config.spread_buffer
            elif self.config.spread_mode == 2:  # SMART
                # TODO: Implement smart spread calculation based on history
                buffer = int(current_spread * 1.2) + 1
            else:  # NONE
                buffer = 0
            
            # Validate buffer
            buffer = max(0, min(buffer, 100))  # Cap between 0-100 points
            
            return buffer
            
        except Exception as e:
            self.logger.error(f"Spread buffer calculation error: {e}")
            return 5  # Default fallback
    
    def check_trading_conditions(self) -> Dict:
        """Check if trading conditions are met with enhanced validation"""
        conditions = {
            "can_trade": False,
            "reason": "",
            "details": {}
        }
        
        try:
            # Check connection
            if not self.is_connected or not self.is_validated:
                conditions["reason"] = "MT5 not connected or validated"
                return conditions
            
            # Check if trading is enabled
            if self.config.trading_direction == 3:  # STOP
                conditions["reason"] = "Trading stopped by user"
                return conditions
            
            # Check account status
            account = mt5.account_info()
            if account is None:
                conditions["reason"] = "Cannot get account information"
                return conditions
            
            if not account.trade_allowed:
                conditions["reason"] = "Trading not allowed on account"
                return conditions
            
            # Check margin level
            if account.margin > 0:
                margin_level = account.equity / account.margin * 100
                if margin_level < 100:
                    conditions["reason"] = f"Low margin level: {margin_level:.1f}%"
                    return conditions
            
            # Check spread
            current_spread = self.get_current_spread()
            if current_spread > self.config.max_spread_alert:
                conditions["reason"] = f"Spread too high: {current_spread} points"
                conditions["details"]["spread"] = current_spread
                return conditions
            
            # Check symbol trading
            symbol = mt5.symbol_info(self.config.symbol)
            if symbol is None:
                conditions["reason"] = "Symbol information not available"
                return conditions
            
            if symbol.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                conditions["reason"] = "Trading disabled for symbol"
                return conditions
            
            # Check market hours (basic check)
            if not self._is_market_open():
                conditions["reason"] = "Market closed"
                return conditions
            
            # All checks passed
            conditions["can_trade"] = True
            conditions["details"] = {
                "spread": current_spread,
                "margin_level": margin_level if account.margin > 0 else 999.99,
                "balance": account.balance,
                "equity": account.equity
            }
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Trading conditions check error: {e}")
            conditions["reason"] = f"Error checking conditions: {e}"
            return conditions
    
    def _is_market_open(self) -> bool:
        """Basic market hours check"""
        try:
            # Get current server time
            now = datetime.now()
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # Basic forex hours: Monday 00:00 - Friday 23:59 (server time)
            if weekday == 6:  # Sunday - limited hours
                return now.hour >= 21  # After 9 PM
            elif weekday == 5:  # Saturday 
                return now.hour < 1   # Before 1 AM
            else:  # Monday-Friday
                return True
            
        except Exception as e:
            self.logger.error(f"Market hours check error: {e}")
            return True  # Default to open if error
    
    def analyze_entry_signals(self) -> Dict:
        """Analyze entry signals based on Fractal + RSI with validation"""
        if not self.is_connected:
            return {"error": "MT5 not connected"}
        
        try:
            data = self.get_market_data()
            if data is None or len(data) < 50:
                return {"error": "Insufficient market data", "bars": len(data) if data is not None else 0}
            
            # Calculate indicators
            rsi = self.calculate_rsi(data)
            fractal_up, fractal_down = self.find_fractals(data)
            
            if rsi.empty or fractal_up.empty or fractal_down.empty:
                return {"error": "Indicator calculation failed"}
            
            current_rsi = rsi.iloc[-1]
            
            # Check for recent fractals (within last N bars)
            lookback = self.config.fractal_period
            latest_fractal_up = fractal_up.iloc[-lookback:].any()
            latest_fractal_down = fractal_down.iloc[-lookback:].any()
            
            signals = {}
            
            # BUY Signal: Fractal Down + RSI > RSI_UP
            if latest_fractal_down and current_rsi > self.config.rsi_up:
                if self.config.trading_direction in [0, 1]:  # BOTH or BUY_ONLY
                    signals["BUY"] = {
                        "rsi": current_rsi,
                        "rsi_threshold": self.config.rsi_up,
                        "fractal_down": True,
                        "strength": min(100, (current_rsi - self.config.rsi_up) * 2),
                        "timestamp": datetime.now()
                    }
            
            # SELL Signal: Fractal Up + RSI < RSI_DOWN
            if latest_fractal_up and current_rsi < self.config.rsi_down:
                if self.config.trading_direction in [0, 2]:  # BOTH or SELL_ONLY
                    signals["SELL"] = {
                        "rsi": current_rsi,
                        "rsi_threshold": self.config.rsi_down,
                        "fractal_up": True,
                        "strength": min(100, (self.config.rsi_down - current_rsi) * 2),
                        "timestamp": datetime.now()
                    }
            
            # Add current market info
            signals["market_info"] = {
                "current_rsi": current_rsi,
                "spread": self.get_current_spread(),
                "price": data['close'].iloc[-1],
                "time": data['time'].iloc[-1],
                "bars_analyzed": len(data)
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal analysis error: {e}")
            return {"error": f"Signal analysis failed: {e}"}
    
    def disconnect(self):
        """Properly disconnect from MT5"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                self.is_validated = False
                self.logger.info("MT5 disconnected")
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")