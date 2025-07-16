import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import json
from enum import Enum
from trading_core import TradingConfig
from position_manager import PositionManager
from order_executor import OrderExecutor

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TradingSession(Enum):
    ASIAN = "asian"
    EUROPEAN = "european"
    AMERICAN = "american"
    OVERLAP = "overlap"

@dataclass
class RiskMetrics:
    """Enhanced risk metrics tracking"""
    # Real-time account data
    current_balance: float = 0.0
    current_equity: float = 0.0
    current_margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    
    # P&L tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    
    # Drawdown tracking
    peak_balance: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    balance_drawdown: float = 0.0
    equity_drawdown: float = 0.0
    
    # Trading performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Risk ratios
    risk_reward_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    exposure: float = 0.0  # % of balance in open positions
    used_margin_percent: float = 0.0
    
    # Validation flags
    data_valid: bool = False
    last_update: datetime = None

@dataclass
class RiskLimits:
    """Enhanced risk limits configuration"""
    # Loss limits
    daily_loss_limit: float = 100.0
    weekly_loss_limit: float = 500.0
    monthly_loss_limit: float = 2000.0
    
    # Drawdown limits
    max_drawdown_percent: float = 10.0
    max_balance_drawdown: float = 15.0
    max_equity_drawdown: float = 20.0
    
    # Position limits
    max_positions: int = 5
    max_lot_size: float = 1.0
    max_total_volume: float = 5.0
    
    # Margin limits
    min_margin_level: float = 200.0  # %
    max_used_margin: float = 50.0    # %
    emergency_margin_level: float = 100.0  # %
    
    # Account limits
    min_account_balance: float = 100.0  # Reduced for testing
    min_free_margin: float = 50.0
    
    # Market limits
    max_spread_trading: int = 50    # Increased for testing
    max_slippage_points: int = 30
    
    # Exposure limits
    max_exposure_percent: float = 30.0
    max_correlation: float = 0.8

@dataclass
class MarketCondition:
    """Market condition assessment"""
    volatility: float = 0.0
    trend_strength: float = 0.0
    session: TradingSession = TradingSession.ASIAN
    news_risk: bool = False
    weekend_gap: bool = False
    low_liquidity: bool = False
    high_spread: bool = False
    market_closed: bool = False

class RiskManager:
    def __init__(self, config: TradingConfig, position_manager: PositionManager):
        self.config = config
        self.position_manager = position_manager
        self.risk_limits = RiskLimits()
        self.risk_metrics = RiskMetrics()
        
        # Trading restrictions
        self.trading_allowed = True
        self.risk_level = RiskLevel.LOW
        self.current_restrictions = []
        self.last_restriction_check = None
        
        # Market condition tracking
        self.market_condition = MarketCondition()
        
        # Historical data for calculations
        self.balance_history = []
        self.equity_history = []
        self.daily_snapshots = []
        
        # Performance tracking
        self.trade_history = []
        self.deals_processed = set()  # Track processed deals
        
        # Account info tracking
        self.account_currency = "USD"
        self.account_leverage = 0
        self.account_server = ""
        
        # News and event tracking
        self.high_impact_news_times = []
        self.trading_sessions = self._init_trading_sessions()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with current account info
        self._initialize_account_baseline()
    
    def _init_trading_sessions(self) -> Dict[TradingSession, Dict]:
        """Initialize trading session times (UTC)"""
        return {
            TradingSession.ASIAN: {"start": time(0, 0), "end": time(9, 0)},
            TradingSession.EUROPEAN: {"start": time(7, 0), "end": time(16, 0)},
            TradingSession.AMERICAN: {"start": time(13, 0), "end": time(22, 0)}
        }
    
    def _initialize_account_baseline(self):
        """Initialize account baseline data with better error handling"""
        try:
            self.logger.info("Initializing risk manager baseline...")
            
            # Get initial account data
            if self._update_account_data():
                # Set initial peaks
                self.risk_metrics.peak_balance = self.risk_metrics.current_balance
                self.risk_metrics.peak_equity = self.risk_metrics.current_equity
                
                # Create initial snapshot
                self._create_daily_snapshot()
                
                self.logger.info(f"✓ Risk manager initialized:")
                self.logger.info(f"  Initial balance: ${self.risk_metrics.current_balance:.2f}")
                self.logger.info(f"  Initial equity: ${self.risk_metrics.current_equity:.2f}")
                self.logger.info(f"  Free margin: ${self.risk_metrics.free_margin:.2f}")
                self.logger.info(f"  Margin level: {self.risk_metrics.margin_level:.2f}%")
                self.logger.info(f"  Account currency: {self.account_currency}")
                self.logger.info(f"  Leverage: 1:{self.account_leverage}")
            else:
                self.logger.warning("Could not get initial account data, using defaults")
                self._set_default_values()
                
        except Exception as e:
            self.logger.error(f"Risk manager initialization error: {e}")
            self._set_default_values()

    def _set_default_values(self):
        """Set default values when account data unavailable"""
        try:
            self.risk_metrics.current_balance = 1000.0
            self.risk_metrics.current_equity = 1000.0
            self.risk_metrics.peak_balance = 1000.0
            self.risk_metrics.peak_equity = 1000.0
            self.risk_metrics.current_margin = 0.0
            self.risk_metrics.free_margin = 1000.0
            self.risk_metrics.margin_level = 999.99
            self.risk_metrics.used_margin_percent = 0.0
            self.account_currency = "USD"
            self.account_leverage = 100
            self.risk_metrics.data_valid = False
            
            self.logger.info("Using default account values for testing")
            
        except Exception as e:
            self.logger.error(f"Set default values error: {e}")    
    
    def update_risk_limits(self, new_limits: Dict):
        """Update risk limits from UI with validation"""
        try:
            updated_count = 0
            for key, value in new_limits.items():
                if hasattr(self.risk_limits, key):
                    # Validate values
                    if self._validate_risk_limit(key, value):
                        old_value = getattr(self.risk_limits, key)
                        setattr(self.risk_limits, key, value)
                        if old_value != value:
                            self.logger.info(f"Risk limit updated: {key} = {value}")
                            updated_count += 1
                    else:
                        self.logger.warning(f"Invalid risk limit value: {key} = {value}")
            
            if updated_count > 0:
                self.logger.info(f"Updated {updated_count} risk limits")
                
        except Exception as e:
            self.logger.error(f"Risk limits update error: {e}")
    
    def _validate_risk_limit(self, key: str, value) -> bool:
        """Validate risk limit values"""
        try:
            # Basic type checking
            if not isinstance(value, (int, float)):
                return False
            
            # Specific validations
            validations = {
                'daily_loss_limit': lambda x: 0 < x <= 10000,
                'weekly_loss_limit': lambda x: 0 < x <= 50000,
                'monthly_loss_limit': lambda x: 0 < x <= 200000,
                'max_drawdown_percent': lambda x: 0 < x <= 50,
                'max_positions': lambda x: 1 <= x <= 50,
                'max_lot_size': lambda x: 0.01 <= x <= 100,
                'min_margin_level': lambda x: 50 <= x <= 1000,
                'max_spread_trading': lambda x: 1 <= x <= 200,
                'min_account_balance': lambda x: 1 <= x <= 100000
            }
            
            validator = validations.get(key)
            return validator(value) if validator else True
            
        except Exception as e:
            self.logger.error(f"Risk limit validation error: {e}")
            return False
    
    def check_trading_allowed(self) -> Tuple[bool, List[str]]:
        """Enhanced trading permission check"""
        try:
            restrictions = []
            
            # Update all metrics first
            self.update_metrics()
            
            # Check account data validity
            if not self.risk_metrics.data_valid:
                restrictions.append("Account data unavailable or invalid")
            
            # Check critical account limits
            if self.risk_metrics.current_balance < self.risk_limits.min_account_balance:
                restrictions.append(f"Balance too low: ${self.risk_metrics.current_balance:.2f} < ${self.risk_limits.min_account_balance:.2f}")
            
            # Check margin level
            if self.risk_metrics.margin_level < self.risk_limits.min_margin_level:
                restrictions.append(f"Margin level too low: {self.risk_metrics.margin_level:.1f}% < {self.risk_limits.min_margin_level:.1f}%")
            
            # Emergency margin level check
            if self.risk_metrics.margin_level < self.risk_limits.emergency_margin_level:
                restrictions.append(f"EMERGENCY: Margin level critical: {self.risk_metrics.margin_level:.1f}%")
                self.risk_level = RiskLevel.CRITICAL
            
            # Check daily loss limit
            if self.risk_metrics.daily_pnl <= -self.risk_limits.daily_loss_limit:
                restrictions.append(f"Daily loss limit: ${abs(self.risk_metrics.daily_pnl):.2f} >= ${self.risk_limits.daily_loss_limit:.2f}")
            
            # Check weekly loss limit
            if self.risk_metrics.weekly_pnl <= -self.risk_limits.weekly_loss_limit:
                restrictions.append(f"Weekly loss limit: ${abs(self.risk_metrics.weekly_pnl):.2f} >= ${self.risk_limits.weekly_loss_limit:.2f}")
            
            # Check monthly loss limit
            if self.risk_metrics.monthly_pnl <= -self.risk_limits.monthly_loss_limit:
                restrictions.append(f"Monthly loss limit: ${abs(self.risk_metrics.monthly_pnl):.2f} >= ${self.risk_limits.monthly_loss_limit:.2f}")
            
            # Check drawdown limits
            if self.risk_metrics.current_drawdown >= self.risk_limits.max_drawdown_percent:
                restrictions.append(f"Max drawdown: {self.risk_metrics.current_drawdown:.2f}% >= {self.risk_limits.max_drawdown_percent:.2f}%")
            
            # Check equity drawdown
            if self.risk_metrics.equity_drawdown >= self.risk_limits.max_equity_drawdown:
                restrictions.append(f"Equity drawdown: {self.risk_metrics.equity_drawdown:.2f}% >= {self.risk_limits.max_equity_drawdown:.2f}%")
            
            # Check position limits
            active_positions = len(self.position_manager.positions)
            if active_positions >= self.risk_limits.max_positions:
                restrictions.append(f"Max positions: {active_positions} >= {self.risk_limits.max_positions}")
            
            # Check total volume
            total_volume = sum(pos.volume for pos in self.position_manager.positions.values())
            if total_volume >= self.risk_limits.max_total_volume:
                restrictions.append(f"Max total volume: {total_volume:.2f} >= {self.risk_limits.max_total_volume:.2f}")
            
            # Check used margin percentage
            if self.risk_metrics.used_margin_percent >= self.risk_limits.max_used_margin:
                restrictions.append(f"Max used margin: {self.risk_metrics.used_margin_percent:.1f}% >= {self.risk_limits.max_used_margin:.1f}%")
            
            # Check market conditions
            market_restrictions = self._check_market_conditions()
            restrictions.extend(market_restrictions)
            
            # Check news events
            if self._is_high_impact_news_time():
                restrictions.append("High impact news event")
            
            # Update state
            self.current_restrictions = restrictions
            self.trading_allowed = len(restrictions) == 0
            self.last_restriction_check = datetime.now()
            
            # Log restrictions if any
            if restrictions:
                self.logger.warning(f"Trading restricted: {len(restrictions)} restrictions")
                for restriction in restrictions[:3]:  # Log first 3
                    self.logger.warning(f"  • {restriction}")
            
            return self.trading_allowed, restrictions
            
        except Exception as e:
            self.logger.error(f"Trading permission check error: {e}")
            return False, [f"Error checking permissions: {e}"]
    
    def _update_account_data(self) -> bool:
        """Update account data from MT5 with safe attribute access"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.warning("Could not get account info from MT5")
                return False
            
            # Update basic account info with safe attribute access
            self.account_currency = getattr(account_info, 'currency', 'USD')
            self.account_leverage = getattr(account_info, 'leverage', 100)
            self.account_server = getattr(account_info, 'server', '')
            
            # Update risk metrics with safe access
            balance = getattr(account_info, 'balance', 0.0)
            equity = getattr(account_info, 'equity', 0.0)
            margin = getattr(account_info, 'margin', 0.0)
            
            # Handle free_margin - calculate if not available
            free_margin = getattr(account_info, 'free_margin', None)
            if free_margin is None:
                # Calculate free margin: equity - used margin
                free_margin = equity - margin if equity >= margin else equity
            
            # Update risk metrics
            self.risk_metrics.current_balance = float(balance)
            self.risk_metrics.current_equity = float(equity)
            self.risk_metrics.current_margin = float(margin)
            self.risk_metrics.free_margin = float(free_margin)
            
            # Calculate margin level safely
            if self.risk_metrics.current_margin > 0:
                self.risk_metrics.margin_level = (self.risk_metrics.current_equity / self.risk_metrics.current_margin) * 100
            else:
                self.risk_metrics.margin_level = 999.99  # No positions = infinite margin level
            
            # Calculate used margin percentage
            if self.risk_metrics.current_equity > 0:
                self.risk_metrics.used_margin_percent = (self.risk_metrics.current_margin / self.risk_metrics.current_equity) * 100
            else:
                self.risk_metrics.used_margin_percent = 0
            
            # Update peaks
            if self.risk_metrics.current_balance > self.risk_metrics.peak_balance:
                self.risk_metrics.peak_balance = self.risk_metrics.current_balance
            
            if self.risk_metrics.current_equity > self.risk_metrics.peak_equity:
                self.risk_metrics.peak_equity = self.risk_metrics.current_equity
            
            # Add to history
            timestamp = datetime.now()
            self.balance_history.append({
                "timestamp": timestamp,
                "balance": self.risk_metrics.current_balance,
                "equity": self.risk_metrics.current_equity,
                "margin": self.risk_metrics.current_margin,
                "free_margin": self.risk_metrics.free_margin,
                "margin_level": self.risk_metrics.margin_level
            })
            
            # Keep only last 1000 records
            if len(self.balance_history) > 1000:
                self.balance_history = self.balance_history[-1000:]
            
            self.risk_metrics.data_valid = True
            self.risk_metrics.last_update = timestamp
            
            return True
            
        except Exception as e:
            self.logger.error(f"Account data update error: {e}")
            self.risk_metrics.data_valid = False
            return False
        
    def update_metrics(self):
        """Update all risk metrics"""
        try:
            # Update account data
            self._update_account_data()
            
            # Update P&L metrics
            self._update_pnl_metrics()
            
            # Update drawdown metrics
            self._update_drawdown_metrics()
            
            # Update trade performance metrics
            self._update_trade_metrics()
            
            # Update market condition
            self._update_market_condition()
            
            # Assess current risk level
            self.assess_risk_level()
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")
    
    def _update_pnl_metrics(self):
        """Update P&L metrics for different periods"""
        try:
            if not self.balance_history:
                return
            
            now = datetime.now()
            
            # Daily P&L - from start of day
            daily_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            daily_records = [r for r in self.balance_history if r["timestamp"] >= daily_start]
            
            if daily_records:
                day_start_balance = daily_records[0]["balance"]
                self.risk_metrics.daily_pnl = self.risk_metrics.current_balance - day_start_balance
            elif self.daily_snapshots:
                # Use daily snapshot if available
                today_snapshot = None
                for snapshot in reversed(self.daily_snapshots):
                    if snapshot["date"] == now.date():
                        today_snapshot = snapshot
                        break
                
                if today_snapshot:
                    self.risk_metrics.daily_pnl = self.risk_metrics.current_balance - today_snapshot["balance"]
            
            # Weekly P&L
            weekly_start = now - timedelta(days=7)
            weekly_records = [r for r in self.balance_history if r["timestamp"] >= weekly_start]
            if weekly_records:
                self.risk_metrics.weekly_pnl = self.risk_metrics.current_balance - weekly_records[0]["balance"]
            
            # Monthly P&L
            monthly_start = now - timedelta(days=30)
            monthly_records = [r for r in self.balance_history if r["timestamp"] >= monthly_start]
            if monthly_records:
                self.risk_metrics.monthly_pnl = self.risk_metrics.current_balance - monthly_records[0]["balance"]
                
        except Exception as e:
            self.logger.error(f"P&L metrics update error: {e}")
    
    def _update_drawdown_metrics(self):
        """Update drawdown metrics"""
        try:
            # Balance drawdown
            if self.risk_metrics.peak_balance > 0:
                balance_dd_amount = self.risk_metrics.peak_balance - self.risk_metrics.current_balance
                self.risk_metrics.balance_drawdown = (balance_dd_amount / self.risk_metrics.peak_balance) * 100
            
            # Equity drawdown  
            if self.risk_metrics.peak_equity > 0:
                equity_dd_amount = self.risk_metrics.peak_equity - self.risk_metrics.current_equity
                self.risk_metrics.equity_drawdown = (equity_dd_amount / self.risk_metrics.peak_equity) * 100
            
            # Current drawdown (use worse of balance/equity)
            self.risk_metrics.current_drawdown = max(
                self.risk_metrics.balance_drawdown,
                self.risk_metrics.equity_drawdown
            )
            
            # Update max drawdown
            if self.risk_metrics.current_drawdown > self.risk_metrics.max_drawdown:
                self.risk_metrics.max_drawdown = self.risk_metrics.current_drawdown
                
        except Exception as e:
            self.logger.error(f"Drawdown metrics update error: {e}")
    
    def _update_trade_metrics(self):
        """Update trading performance metrics"""
        try:
            # Get recent deals
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=1),  # Last 24 hours
                datetime.now()
            )
            
            if deals:
                # Filter for our symbol and new deals
                symbol_deals = [
                    deal for deal in deals 
                    if deal.symbol == self.config.symbol 
                    and deal.profit != 0 
                    and deal.ticket not in self.deals_processed
                ]
                
                # Process new deals
                profits = []
                for deal in symbol_deals:
                    profits.append(deal.profit)
                    self.deals_processed.add(deal.ticket)
                    
                    # Add to trade history
                    self.trade_history.append({
                        "ticket": deal.ticket,
                        "time": datetime.fromtimestamp(deal.time),
                        "profit": deal.profit,
                        "volume": deal.volume,
                        "price": deal.price
                    })
                
                # Keep trade history manageable
                if len(self.trade_history) > 1000:
                    self.trade_history = self.trade_history[-1000:]
                
                # Update metrics if we have trades
                if profits:
                    self.risk_metrics.total_trades = len(self.trade_history)
                    
                    # Calculate win/loss
                    wins = [p for p in profits if p > 0]
                    losses = [p for p in profits if p < 0]
                    
                    self.risk_metrics.winning_trades = len(wins)
                    self.risk_metrics.losing_trades = len(losses)
                    
                    # Calculate win rate
                    if self.risk_metrics.total_trades > 0:
                        self.risk_metrics.win_rate = (self.risk_metrics.winning_trades / self.risk_metrics.total_trades) * 100
                    
                    # Calculate average win/loss
                    self.risk_metrics.avg_win = sum(wins) / len(wins) if wins else 0
                    self.risk_metrics.avg_loss = sum(losses) / len(losses) if losses else 0
                    
                    # Calculate profit factor
                    gross_profit = sum(wins)
                    gross_loss = abs(sum(losses))
                    
                    if gross_loss > 0:
                        self.risk_metrics.profit_factor = gross_profit / gross_loss
                    else:
                        self.risk_metrics.profit_factor = 999.99 if gross_profit > 0 else 0
                        
        except Exception as e:
            self.logger.error(f"Trade metrics update error: {e}")
    
    def _create_daily_snapshot(self):
        """Create daily snapshot for P&L calculation"""
        try:
            today = datetime.now().date()
            
            # Check if snapshot already exists for today
            existing = [s for s in self.daily_snapshots if s["date"] == today]
            if existing:
                return  # Already have today's snapshot
            
            snapshot = {
                "date": today,
                "balance": self.risk_metrics.current_balance,
                "equity": self.risk_metrics.current_equity,
                "timestamp": datetime.now()
            }
            
            self.daily_snapshots.append(snapshot)
            
            # Keep only last 90 days
            cutoff_date = today - timedelta(days=90)
            self.daily_snapshots = [
                s for s in self.daily_snapshots 
                if s["date"] >= cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Daily snapshot error: {e}")
    
    def validate_order_size(self, volume: float, order_type: str) -> Tuple[bool, float, str]:
        """Enhanced order size validation"""
        try:
            original_volume = volume
            
            # Basic volume validation
            if volume <= 0:
                return False, 0, "Volume must be positive"
            
            # Check maximum lot size
            if volume > self.risk_limits.max_lot_size:
                volume = self.risk_limits.max_lot_size
                self.logger.warning(f"Volume reduced to max limit: {volume}")
            
            # Check total volume limit
            current_total_volume = sum(pos.volume for pos in self.position_manager.positions.values())
            if current_total_volume + volume > self.risk_limits.max_total_volume:
                max_additional = self.risk_limits.max_total_volume - current_total_volume
                if max_additional <= 0:
                    return False, 0, "Total volume limit reached"
                volume = min(volume, max_additional)
            
            # Check margin requirements
            symbol_info = mt5.symbol_info(self.config.symbol)
            if symbol_info and self.risk_metrics.current_equity > 0:
                required_margin = volume * symbol_info.margin_initial
                
                # Check if we have enough free margin
                if required_margin > self.risk_metrics.free_margin:
                    max_volume_by_margin = self.risk_metrics.free_margin / symbol_info.margin_initial
                    if max_volume_by_margin < symbol_info.volume_min:
                        return False, 0, "Insufficient margin"
                    volume = min(volume, max_volume_by_margin)
                
                # Check used margin percentage after trade
                new_used_margin = ((self.risk_metrics.current_margin + required_margin) / self.risk_metrics.current_equity) * 100
                if new_used_margin > self.risk_limits.max_used_margin:
                    return False, 0, f"Would exceed max margin usage: {new_used_margin:.1f}%"
            
            # Validate against symbol specifications
            if symbol_info:
                # Check minimum volume
                if volume < symbol_info.volume_min:
                    return False, 0, f"Volume below minimum: {symbol_info.volume_min}"
                
                # Check maximum volume
                if volume > symbol_info.volume_max:
                    volume = symbol_info.volume_max
                
                # Round to valid step
                volume_step = symbol_info.volume_step
                volume = round(volume / volume_step) * volume_step
            
            # Final validation
            if volume <= 0:
                return False, 0, "Calculated volume is zero or negative"
            
            if volume != original_volume:
                message = f"Volume adjusted from {original_volume} to {volume}"
            else:
                message = "Volume validated"
            
            return True, volume, message
            
        except Exception as e:
            self.logger.error(f"Order size validation error: {e}")
            return False, 0, f"Validation error: {e}"
    
    def assess_risk_level(self) -> RiskLevel:
        """Enhanced risk level assessment"""
        try:
            risk_score = 0
            
            # Account health (40% weight)
            if self.risk_metrics.margin_level < self.risk_limits.emergency_margin_level:
                risk_score += 40  # Critical
            elif self.risk_metrics.margin_level < self.risk_limits.min_margin_level:
                risk_score += 25  # High
            elif self.risk_metrics.margin_level < self.risk_limits.min_margin_level * 1.5:
                risk_score += 10  # Medium
            
            # Drawdown risk (30% weight)
            dd_ratio = self.risk_metrics.current_drawdown / self.risk_limits.max_drawdown_percent
            if dd_ratio >= 1.0:
                risk_score += 30  # Critical
            elif dd_ratio >= 0.8:
                risk_score += 20  # High
            elif dd_ratio >= 0.5:
                risk_score += 10  # Medium
            elif dd_ratio >= 0.3:
                risk_score += 5   # Low-Medium
            
            # Daily loss risk (20% weight)
            if self.risk_metrics.daily_pnl < 0:
                loss_ratio = abs(self.risk_metrics.daily_pnl) / self.risk_limits.daily_loss_limit
                if loss_ratio >= 1.0:
                    risk_score += 20  # Critical
                elif loss_ratio >= 0.8:
                    risk_score += 15  # High
                elif loss_ratio >= 0.5:
                    risk_score += 8   # Medium
                elif loss_ratio >= 0.3:
                    risk_score += 3   # Low-Medium
            
            # Position concentration risk (10% weight)
            position_count = len(self.position_manager.positions)
            position_ratio = position_count / self.risk_limits.max_positions
            if position_ratio >= 1.0:
                risk_score += 10
            elif position_ratio >= 0.8:
                risk_score += 6
            elif position_ratio >= 0.6:
                risk_score += 3
            
            # Determine risk level based on score
            if risk_score >= 70:
                self.risk_level = RiskLevel.CRITICAL
            elif risk_score >= 40:
                self.risk_level = RiskLevel.HIGH
            elif risk_score >= 20:
                self.risk_level = RiskLevel.MEDIUM
            else:
                self.risk_level = RiskLevel.LOW
            
            return self.risk_level
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return RiskLevel.MEDIUM  # Default to medium on error
    
    def _check_market_conditions(self) -> List[str]:
        """Enhanced market conditions check"""
        restrictions = []
        
        try:
            # Check spread
            current_spread = self._get_current_spread()
            if current_spread > self.risk_limits.max_spread_trading:
                restrictions.append(f"Spread too high: {current_spread} > {self.risk_limits.max_spread_trading} points")
                self.market_condition.high_spread = True
            else:
                self.market_condition.high_spread = False
            
            # Check market hours
            if not self._is_market_open():
                restrictions.append("Market closed")
                self.market_condition.market_closed = True
            else:
                self.market_condition.market_closed = False
            
            # Check volatility
            volatility = self._calculate_volatility()
            self.market_condition.volatility = volatility
            if volatility > 3.0:  # High volatility threshold
                restrictions.append(f"High volatility: {volatility:.2f}")
            
            # Check weekend gap
            if self._is_weekend_gap():
                restrictions.append("Weekend gap detected")
                self.market_condition.weekend_gap = True
            else:
                self.market_condition.weekend_gap = False
            
            # Check low liquidity
            if self._is_low_liquidity_period():
                restrictions.append("Low liquidity period")
                self.market_condition.low_liquidity = True
            else:
                self.market_condition.low_liquidity = False
            
        except Exception as e:
            self.logger.error(f"Market conditions check error: {e}")
            restrictions.append(f"Market check error: {e}")
        
        return restrictions
    
    def _get_current_spread(self) -> float:
        """Get current spread with error handling"""
        try:
            symbol_info = mt5.symbol_info(self.config.symbol)
            return symbol_info.spread if symbol_info else 30  # Default fallback
        except:
            return 30
    
    def _is_market_open(self) -> bool:
        """Enhanced market hours check"""
        try:
            now = datetime.now()
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            hour = now.hour
            
            # Market closed periods
            if weekday == 6 and hour < 21:  # Sunday before 9 PM
                return False
            elif weekday == 5 and hour > 23:  # Friday after 11 PM
                return False
            elif weekday == 6 and hour >= 0:  # Saturday
                return False
            
            # Check for symbols that trade 24/5
            return True  # XAUUSD typically trades 24/5
            
        except Exception as e:
            self.logger.error(f"Market hours check error: {e}")
            return True  # Default to open on error
    
    def _calculate_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_H1, 0, 24)
            if rates is None or len(rates) < 14:
                return 0.5
            
            df = pd.DataFrame(rates)
            
            # Calculate ATR
            df['hl'] = df['high'] - df['low']
            df['hc'] = abs(df['high'] - df['close'].shift(1))
            df['lc'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
            
            atr = df['tr'].rolling(window=14).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            return (atr / current_price) * 100
            
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {e}")
            return 0.5
    
    def _is_weekend_gap(self) -> bool:
        """Detect weekend gap"""
        try:
            rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_H1, 0, 5)
            if rates is None or len(rates) < 2:
                return False
            
            last_close = rates[-2]['close']
            current_open = rates[-1]['open']
            gap_size = abs(current_open - last_close)
            gap_threshold = last_close * 0.005  # 0.5% gap threshold
            
            return gap_size > gap_threshold
        except:
            return False
    
    def _is_low_liquidity_period(self) -> bool:
        """Check for low liquidity periods"""
        try:
            current_time = datetime.utcnow().time()
            
            # Low liquidity periods (UTC)
            low_periods = [
                (time(22, 0), time(23, 59)),  # After NY close
                (time(0, 0), time(1, 0)),     # Weekend transition
            ]
            
            for start, end in low_periods:
                if start <= current_time <= end:
                    return True
            return False
        except:
            return False
    
    def _is_high_impact_news_time(self) -> bool:
        """Check for high impact news"""
        current_time = datetime.now()
        
        for news_time in self.high_impact_news_times:
            if abs((current_time - news_time).total_seconds()) < 1800:  # 30 min window
                return True
        return False
    
    def _update_market_condition(self):
        """Update market condition assessment"""
        try:
            # Update trading session
            current_time = datetime.utcnow().time()
            self.market_condition.session = TradingSession.ASIAN  # Default
            
            for session, times in self.trading_sessions.items():
                if times["start"] <= current_time <= times["end"]:
                    self.market_condition.session = session
                    break
            
            # Update volatility and trend
            self.market_condition.volatility = self._calculate_volatility()
            self.market_condition.trend_strength = self._calculate_trend_strength()
            
        except Exception as e:
            self.logger.error(f"Market condition update error: {e}")
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength"""
        try:
            rates = mt5.copy_rates_from_pos(self.config.symbol, mt5.TIMEFRAME_H4, 0, 50)
            if rates is None or len(rates) < 20:
                return 0.5
            
            df = pd.DataFrame(rates)
            close_prices = df['close']
            sma_short = close_prices.rolling(window=10).mean()
            sma_long = close_prices.rolling(window=20).mean()
            
            trend_diff = abs(sma_short.iloc[-1] - sma_long.iloc[-1])
            trend_strength = (trend_diff / close_prices.iloc[-1]) * 100
            
            return min(trend_strength, 100)
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return 0.5
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        try:
            return {
                "risk_level": self.risk_level.value,
                "trading_allowed": self.trading_allowed,
                "restrictions": self.current_restrictions,
                "data_valid": self.risk_metrics.data_valid,
                "last_update": self.risk_metrics.last_update.isoformat() if self.risk_metrics.last_update else None,
                
                "account": {
                    "balance": self.risk_metrics.current_balance,
                    "equity": self.risk_metrics.current_equity,
                    "margin": self.risk_metrics.current_margin,
                    "free_margin": self.risk_metrics.free_margin,
                    "margin_level": self.risk_metrics.margin_level,
                    "used_margin_percent": self.risk_metrics.used_margin_percent,
                    "currency": self.account_currency,
                    "leverage": self.account_leverage
                },
                
                "pnl": {
                    "daily_pnl": self.risk_metrics.daily_pnl,
                    "weekly_pnl": self.risk_metrics.weekly_pnl,
                    "monthly_pnl": self.risk_metrics.monthly_pnl
                },
                
                "drawdown": {
                    "current_drawdown": self.risk_metrics.current_drawdown,
                    "max_drawdown": self.risk_metrics.max_drawdown,
                    "balance_drawdown": self.risk_metrics.balance_drawdown,
                    "equity_drawdown": self.risk_metrics.equity_drawdown,
                    "peak_balance": self.risk_metrics.peak_balance,
                    "peak_equity": self.risk_metrics.peak_equity
                },
                
                "performance": {
                    "total_trades": self.risk_metrics.total_trades,
                    "winning_trades": self.risk_metrics.winning_trades,
                    "losing_trades": self.risk_metrics.losing_trades,
                    "win_rate": self.risk_metrics.win_rate,
                    "profit_factor": self.risk_metrics.profit_factor,
                    "avg_win": self.risk_metrics.avg_win,
                    "avg_loss": self.risk_metrics.avg_loss
                },
                
                "market": {
                    "volatility": self.market_condition.volatility,
                    "trend_strength": self.market_condition.trend_strength,
                    "session": self.market_condition.session.value,
                    "high_spread": self.market_condition.high_spread,
                    "low_liquidity": self.market_condition.low_liquidity,
                    "market_closed": self.market_condition.market_closed,
                    "current_spread": self._get_current_spread()
                },
                
                "limits": {
                    "daily_loss_limit": self.risk_limits.daily_loss_limit,
                    "max_drawdown_percent": self.risk_limits.max_drawdown_percent,
                    "max_positions": self.risk_limits.max_positions,
                    "min_margin_level": self.risk_limits.min_margin_level,
                    "max_used_margin": self.risk_limits.max_used_margin
                }
            }
            
        except Exception as e:
            self.logger.error(f"Risk report generation error: {e}")
            return {"error": str(e)}
    
    def emergency_risk_shutdown(self) -> Dict:
        """Emergency shutdown with detailed reporting"""
        try:
            self.trading_allowed = False
            self.risk_level = RiskLevel.CRITICAL
            
            shutdown_reasons = []
            
            # Check all critical conditions
            if self.risk_metrics.margin_level < self.risk_limits.emergency_margin_level:
                shutdown_reasons.append(f"Critical margin level: {self.risk_metrics.margin_level:.1f}%")
            
            if self.risk_metrics.current_drawdown >= self.risk_limits.max_drawdown_percent:
                shutdown_reasons.append(f"Max drawdown exceeded: {self.risk_metrics.current_drawdown:.2f}%")
            
            if abs(self.risk_metrics.daily_pnl) >= self.risk_limits.daily_loss_limit:
                shutdown_reasons.append(f"Daily loss limit: ${abs(self.risk_metrics.daily_pnl):.2f}")
            
            if self.risk_metrics.current_balance < self.risk_limits.min_account_balance:
                shutdown_reasons.append(f"Account balance critical: ${self.risk_metrics.current_balance:.2f}")
            
            self.logger.critical(f"🚨 EMERGENCY RISK SHUTDOWN: {'; '.join(shutdown_reasons)}")
            
            return {
                "shutdown": True,
                "reasons": shutdown_reasons,
                "timestamp": datetime.now().isoformat(),
                "risk_level": self.risk_level.value,
                "account_status": {
                    "balance": self.risk_metrics.current_balance,
                    "equity": self.risk_metrics.current_equity,
                    "margin_level": self.risk_metrics.margin_level,
                    "drawdown": self.risk_metrics.current_drawdown
                }
            }
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")
            return {"shutdown": True, "error": str(e)}