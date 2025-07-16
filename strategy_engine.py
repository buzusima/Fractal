import MetaTrader5 as mt5
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
import queue
from concurrent.futures import ThreadPoolExecutor
import traceback

from trading_core import XAUUSDTradingCore, TradingConfig
from position_manager import PositionManager, Position, RecoveryGroup
from order_executor import OrderExecutor, OrderType, OrderResult
from risk_manager import RiskManager, RiskLevel

class EngineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"

class TradeSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"
    NO_SIGNAL = "NO_SIGNAL"

@dataclass
class EngineStatus:
    """Engine status information"""
    state: EngineState
    uptime: float = 0.0
    last_update: datetime = None
    total_trades: int = 0
    successful_trades: int = 0
    current_positions: int = 0
    total_pnl: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    restrictions: List[str] = None
    last_signal: str = ""
    last_trade: datetime = None
    errors: List[str] = None
    connection_status: str = "unknown"
    reconnection_count: int = 0

@dataclass
class ConnectionHealth:
    """Connection health monitoring"""
    is_connected: bool = False
    last_successful_call: datetime = None
    consecutive_failures: int = 0
    total_reconnections: int = 0
    last_reconnection: datetime = None
    connection_quality: float = 100.0  # 0-100%
    response_time_avg: float = 0.0
    last_error: str = ""

class ThreadSafeQueue:
    """Thread-safe queue for UI updates"""
    def __init__(self, maxsize: int = 1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
    
    def put(self, item, block=True, timeout=None):
        try:
            self.queue.put(item, block=block, timeout=timeout)
        except queue.Full:
            # Remove oldest item if queue is full
            try:
                self.queue.get_nowait()
                self.queue.put(item, block=False)
            except queue.Empty:
                pass
    
    def get(self, block=True, timeout=None):
        return self.queue.get(block=block, timeout=timeout)
    
    def empty(self):
        return self.queue.empty()

class StrategyEngine:
    def __init__(self, config: TradingConfig = None):
        # Initialize components
        self.config = config or TradingConfig()
        self.trading_core = XAUUSDTradingCore(self.config)
        self.position_manager = PositionManager(self.config)
        self.order_executor = OrderExecutor(self.config, self.position_manager)
        self.risk_manager = RiskManager(self.config, self.position_manager)
        
        # Engine state with thread safety
        self.state = EngineState.STOPPED
        self.state_lock = threading.RLock()
        self.start_time = None
        self.last_update = None
        self.running = False
        
        # Connection health monitoring
        self.connection_health = ConnectionHealth()
        self.connection_check_interval = 30.0  # seconds
        self.max_consecutive_failures = 3
        self.reconnection_delay = 5.0  # seconds
        
        # Threading with thread pool
        self.main_thread = None
        self.connection_monitor_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EA_")
        
        # Thread-safe queues for different update types
        self.ui_update_queue = ThreadSafeQueue(100)
        self.trade_event_queue = ThreadSafeQueue(50)
        self.error_queue = ThreadSafeQueue(50)
        
        # Update intervals
        self.update_interval = 1.0  # seconds
        self.signal_check_interval = 5.0  # seconds
        self.position_update_interval = 2.0  # seconds
        self.risk_update_interval = 10.0  # seconds
        
        # Timing tracking
        self.last_signal_check = None
        self.last_position_update = None
        self.last_risk_update = None
        
        # Event handlers with thread safety
        self.event_handlers = {
            'on_trade_opened': [],
            'on_trade_closed': [],
            'on_signal_detected': [],
            'on_risk_alert': [],
            'on_error': [],
            'on_state_changed': [],
            'on_connection_status': []
        }
        self.event_handlers_lock = threading.RLock()
        
        # Performance tracking
        self.engine_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_closed': 0,
            'recovery_triggered': 0,
            'errors_occurred': 0,
            'reconnections': 0,
            'uptime_seconds': 0,
            'avg_loop_time': 0.0,
            'max_loop_time': 0.0
        }
        
        # Signal and trade history
        self.signal_history = []
        self.trade_history = []
        
        # Recovery testing mode
        self.recovery_test_mode = False
        self.recovery_test_results = []
        
        # Smart Recovery Enhancement
        self.recovery_signal_cache = {}  # Store signals for smart recovery
        self.last_successful_signals = {}  # Track last successful signals by type
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection monitoring
        self._setup_connection_monitoring()
    
    def _setup_connection_monitoring(self):
        """Setup connection health monitoring"""
        try:
            self.logger.info("Setting up connection monitoring...")
            
            # Test initial connection
            self._check_connection_health()
            
            self.logger.info("✓ Connection monitoring initialized")
            
        except Exception as e:
            self.logger.error(f"Connection monitoring setup error: {e}")
    
    def start(self) -> bool:
        """Enhanced start with reconnection handling"""
        try:
            with self.state_lock:
                if self.state in [EngineState.RUNNING, EngineState.STARTING]:
                    self.logger.warning("Engine already running or starting")
                    return True
                
                self.logger.info("🚀 Starting XAUUSD Trading Engine...")
                self.state = EngineState.STARTING
                self._notify_state_change()
            
            # Initialize MT5 connection with retries
            if not self._initialize_with_retry():
                with self.state_lock:
                    self.state = EngineState.ERROR
                    self._notify_state_change()
                return False
            
            # Validate configuration
            if not self._validate_config():
                with self.state_lock:
                    self.state = EngineState.ERROR
                    self._notify_state_change()
                return False
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            # Start main trading loop
            self.running = True
            self.start_time = datetime.now()
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True, name="EA_MainLoop")
            self.main_thread.start()
            
            with self.state_lock:
                self.state = EngineState.RUNNING
                self._notify_state_change()
            
            self.logger.info("✅ Trading Engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start engine: {e}")
            self.logger.error(traceback.format_exc())
            with self.state_lock:
                self.state = EngineState.ERROR
                self._notify_state_change()
            return False
    
    def _initialize_with_retry(self, max_retries: int = 3) -> bool:
        """Initialize MT5 with retry logic"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"MT5 initialization attempt {attempt + 1}/{max_retries}")
                
                if self.trading_core.initialize_mt5():
                    self.connection_health.is_connected = True
                    self.connection_health.last_successful_call = datetime.now()
                    self.connection_health.consecutive_failures = 0
                    self.logger.info("✅ MT5 initialized successfully")
                    return True
                else:
                    self.logger.warning(f"❌ MT5 initialization failed (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"MT5 initialization error: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        self.connection_health.is_connected = False
        return False
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        try:
            # Connection monitor thread
            self.connection_monitor_thread = threading.Thread(
                target=self._connection_monitor_loop, 
                daemon=True, 
                name="EA_ConnectionMonitor"
            )
            self.connection_monitor_thread.start()
            
            self.logger.info("✅ Monitoring threads started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring threads: {e}")
    
    def _connection_monitor_loop(self):
        """Background connection monitoring"""
        self.logger.info("Connection monitor started")
        
        while self.running:
            try:
                # Check connection health
                self._check_connection_health()
                
                # Handle reconnection if needed
                if not self.connection_health.is_connected and self.state == EngineState.RUNNING:
                    self._handle_disconnection()
                
                time.sleep(self.connection_check_interval)
                
            except Exception as e:
                self.logger.error(f"Connection monitor error: {e}")
                time.sleep(5)  # Short delay on error
    
    def _check_connection_health(self) -> bool:
        """Check MT5 connection health"""
        try:
            start_time = time.time()
            
            # Test MT5 connection with simple call
            account_info = mt5.account_info()
            symbol_info = mt5.symbol_info(self.config.symbol)
            
            response_time = time.time() - start_time
            
            if account_info is not None and symbol_info is not None:
                # Connection successful
                self.connection_health.is_connected = True
                self.connection_health.last_successful_call = datetime.now()
                self.connection_health.consecutive_failures = 0
                
                # Update response time average
                if self.connection_health.response_time_avg == 0:
                    self.connection_health.response_time_avg = response_time
                else:
                    self.connection_health.response_time_avg = (
                        self.connection_health.response_time_avg * 0.8 + response_time * 0.2
                    )
                
                # Calculate connection quality
                quality = 100.0
                if response_time > 2.0:
                    quality -= min(50, (response_time - 2.0) * 10)
                
                self.connection_health.connection_quality = max(quality, 10.0)
                self.connection_health.last_error = ""
                
                return True
            else:
                # Connection failed
                self._handle_connection_failure("MT5 calls returned None")
                return False
                
        except Exception as e:
            self._handle_connection_failure(f"Connection test error: {e}")
            return False
    
    def _handle_connection_failure(self, error_msg: str):
        """Handle connection failure"""
        self.connection_health.is_connected = False
        self.connection_health.consecutive_failures += 1
        self.connection_health.last_error = error_msg
        self.connection_health.connection_quality = max(
            0, self.connection_health.connection_quality - 20
        )
        
        self.logger.warning(f"Connection failure #{self.connection_health.consecutive_failures}: {error_msg}")
    
    def _handle_disconnection(self):
        """Handle MT5 disconnection with reconnection logic"""
        try:
            if self.connection_health.consecutive_failures < self.max_consecutive_failures:
                return  # Not enough failures yet
            
            with self.state_lock:
                if self.state != EngineState.RUNNING:
                    return  # Already handling
                
                self.state = EngineState.RECONNECTING
                self._notify_state_change()
            
            self.logger.warning("🔄 MT5 disconnected, attempting reconnection...")
            
            # Attempt reconnection
            reconnection_successful = False
            for attempt in range(3):
                self.logger.info(f"Reconnection attempt {attempt + 1}/3")
                
                try:
                    # Shutdown current connection
                    mt5.shutdown()
                    time.sleep(self.reconnection_delay)
                    
                    # Reinitialize
                    if self.trading_core.initialize_mt5():
                        reconnection_successful = True
                        break
                        
                except Exception as e:
                    self.logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
                
                time.sleep(self.reconnection_delay * (attempt + 1))
            
            if reconnection_successful:
                # Successful reconnection
                self.connection_health.is_connected = True
                self.connection_health.consecutive_failures = 0
                self.connection_health.total_reconnections += 1
                self.connection_health.last_reconnection = datetime.now()
                self.engine_stats['reconnections'] += 1
                
                with self.state_lock:
                    self.state = EngineState.RUNNING
                    self._notify_state_change()
                
                self.logger.info("✅ Reconnection successful")
                self._notify_connection_status("reconnected")
                
                # Update all components after reconnection
                self._post_reconnection_update()
                
            else:
                # Reconnection failed
                self.logger.error("❌ All reconnection attempts failed")
                self._notify_connection_status("failed")
                
                with self.state_lock:
                    self.state = EngineState.ERROR
                    self._notify_state_change()
                
                # Trigger emergency stop
                self.emergency_stop()
                
        except Exception as e:
            self.logger.error(f"Reconnection handling error: {e}")
            with self.state_lock:
                self.state = EngineState.ERROR
                self._notify_state_change()
    
    def _post_reconnection_update(self):
        """Update all components after successful reconnection"""
        try:
            self.logger.info("Updating components after reconnection...")
            
            # Update position manager
            self.position_manager.update_positions()
            
            # Update risk manager
            self.risk_manager.update_metrics()
            
            # Reset timing
            self.last_signal_check = None
            self.last_position_update = None
            self.last_risk_update = None
            
            self.logger.info("✅ Post-reconnection update completed")
            
        except Exception as e:
            self.logger.error(f"Post-reconnection update error: {e}")
    
    def stop(self):
        """Enhanced stop with proper cleanup"""
        try:
            self.logger.info("🛑 Stopping Trading Engine...")
            
            with self.state_lock:
                self.running = False
                self.state = EngineState.STOPPED
                self._notify_state_change()
            
            # Wait for threads to finish
            threads_to_join = [
                (self.main_thread, "Main Loop"),
                (self.connection_monitor_thread, "Connection Monitor")
            ]
            
            for thread, name in threads_to_join:
                if thread and thread.is_alive():
                    self.logger.info(f"Waiting for {name} thread...")
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        self.logger.warning(f"{name} thread did not finish gracefully")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True, timeout=10.0)
            
            # Clear queues
            self._clear_queues()
            
            self.logger.info("✅ Trading Engine stopped")
            
        except Exception as e:
            self.logger.error(f"Stop error: {e}")
    
    def _clear_queues(self):
        """Clear all thread-safe queues"""
        try:
            while not self.ui_update_queue.empty():
                self.ui_update_queue.get(block=False)
        except:
            pass
        
        try:
            while not self.trade_event_queue.empty():
                self.trade_event_queue.get(block=False)
        except:
            pass
        
        try:
            while not self.error_queue.empty():
                self.error_queue.get(block=False)
        except:
            pass
    
    def pause(self):
        """Enhanced pause"""
        with self.state_lock:
            if self.state == EngineState.RUNNING:
                self.state = EngineState.PAUSED
                self._notify_state_change()
                self.logger.info("⏸️ Trading Engine paused")
    
    def resume(self):
        """Enhanced resume"""
        with self.state_lock:
            if self.state == EngineState.PAUSED:
                self.state = EngineState.RUNNING
                self._notify_state_change()
                self.logger.info("▶️ Trading Engine resumed")
    
    def emergency_stop(self):
        """Enhanced emergency stop"""
        try:
            self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
            
            with self.state_lock:
                self.state = EngineState.EMERGENCY_STOP
                self._notify_state_change()
            
            # Close all positions in thread pool
            future = self.thread_pool.submit(self._emergency_close_positions)
            
            try:
                closed_positions = future.result(timeout=30.0)
                self.logger.info(f"Emergency closed {len(closed_positions)} positions")
            except Exception as e:
                self.logger.error(f"Emergency close error: {e}")
            
            # Trigger emergency risk shutdown
            self.risk_manager.emergency_risk_shutdown()
            
            # Stop engine
            self.stop()
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
            # Force stop even if error
            self.running = False
    
    def _emergency_close_positions(self) -> List[int]:
        """Emergency close all positions"""
        try:
            return self.position_manager.emergency_close_all()
        except Exception as e:
            self.logger.error(f"Emergency close positions error: {e}")
            return []
    
    def update_config(self, new_config: Dict):
        """Thread-safe configuration update"""
        try:
            with self.state_lock:
                # Update trading core config
                self.trading_core.update_config(new_config)
                
                # Update risk limits if provided
                risk_limits = {k: v for k, v in new_config.items() 
                              if k.startswith(('daily_', 'weekly_', 'monthly_', 'max_', 'min_'))}
                if risk_limits:
                    self.risk_manager.update_risk_limits(risk_limits)
                
                self.logger.info(f"Configuration updated: {len(new_config)} parameters")
                
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            self._notify_error(f"Config update error: {e}")
    
    def _main_loop(self):
        """Enhanced main trading loop with timing optimization"""
        self.logger.info("🔄 Main trading loop started")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Check if we should be running
                with self.state_lock:
                    if not self.running or self.state not in [EngineState.RUNNING, EngineState.PAUSED]:
                        break
                    
                    current_state = self.state
                
                # Only update components if we have connection
                if self.connection_health.is_connected:
                    # Update components with different intervals
                    now = datetime.now()
                    
                    # Update positions (every 2 seconds)
                    if (self.last_position_update is None or 
                        (now - self.last_position_update).total_seconds() >= self.position_update_interval):
                        self._safe_update_positions()
                        self.last_position_update = now
                    
                    # Update risk metrics (every 10 seconds)
                    if (self.last_risk_update is None or 
                        (now - self.last_risk_update).total_seconds() >= self.risk_update_interval):
                        self._safe_update_risk()
                        self.last_risk_update = now
                    
                    # Process trading logic (only when running)
                    if current_state == EngineState.RUNNING:
                        if (self.last_signal_check is None or 
                            (now - self.last_signal_check).total_seconds() >= self.signal_check_interval):
                            self._safe_process_trading_logic()
                            self.last_signal_check = now
                
                # Update engine statistics
                self._update_engine_stats()
                
                # Add status update to queue
                self._queue_status_update()
                
                # Calculate loop timing
                loop_time = time.time() - loop_start
                self._update_loop_timing(loop_time)
                
                # Sleep for remaining interval time
                sleep_time = max(0.1, self.update_interval - loop_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                self._notify_error(f"Main loop error: {e}")
                self.engine_stats['errors_occurred'] += 1
                
                # If too many errors, emergency stop
                if self.engine_stats['errors_occurred'] > 20:
                    self.logger.critical("Too many errors, emergency stopping")
                    self.emergency_stop()
                    break
                
                time.sleep(1.0)  # Prevent tight error loops
        
        self.logger.info("🏁 Main trading loop ended")
    
    def _safe_update_positions(self):
        """Thread-safe position update"""
        try:
            self.position_manager.update_positions()
        except Exception as e:
            self.logger.error(f"Position update error: {e}")
            self._handle_connection_failure(f"Position update failed: {e}")
    
    def _safe_update_risk(self):
        """Thread-safe risk update"""
        try:
            self.risk_manager.update_metrics()
        except Exception as e:
            self.logger.error(f"Risk update error: {e}")
            self._handle_connection_failure(f"Risk update failed: {e}")
    
    def _safe_process_trading_logic(self):
        """Thread-safe trading logic processing"""
        try:
            self._process_trading_logic()
        except Exception as e:
            self.logger.error(f"Trading logic error: {e}")
            self._notify_error(f"Trading logic error: {e}")
    
    def _process_trading_logic(self):
        """Main trading logic processing with enhanced signal handling"""
        # Check if trading is allowed
        trading_allowed, restrictions = self.risk_manager.check_trading_allowed()
        if not trading_allowed:
            return
        
        # Check trading conditions
        conditions = self.trading_core.check_trading_conditions()
        if not conditions.get("can_trade", False):
            return
        
        # Analyze entry signals
        signals = self.trading_core.analyze_entry_signals()
        
        if not signals or 'error' in signals:
            return
        
        # Store signal for smart recovery
        self._cache_current_signals(signals)
        
        # Process signals
        for signal_type, signal_data in signals.items():
            if signal_type in ["BUY", "SELL"]:
                self._process_entry_signal(signal_type, signal_data)
        
        # Check recovery opportunities (enhanced)
        self._check_recovery_opportunities_smart()
        
        # Monitor existing positions
        self._monitor_positions()
    
    def _cache_current_signals(self, signals: Dict):
        """Cache current signals for smart recovery"""
        try:
            timestamp = datetime.now()
            
            for signal_type, signal_data in signals.items():
                if signal_type in ["BUY", "SELL"]:
                    # Cache signal with timestamp
                    self.recovery_signal_cache[signal_type] = {
                        "timestamp": timestamp,
                        "data": signal_data,
                        "strength": signal_data.get("strength", 0),
                        "rsi": signal_data.get("rsi", 50)
                    }
                    
                    # Update last successful signal
                    self.last_successful_signals[signal_type] = {
                        "timestamp": timestamp,
                        "data": signal_data
                    }
            
            # Clean old cached signals (older than 5 minutes)
            cutoff_time = timestamp - timedelta(minutes=5)
            for signal_type in list(self.recovery_signal_cache.keys()):
                if self.recovery_signal_cache[signal_type]["timestamp"] < cutoff_time:
                    del self.recovery_signal_cache[signal_type]
                    
        except Exception as e:
            self.logger.error(f"Signal caching error: {e}")
    
    def _process_entry_signal(self, signal_type: str, signal_data: Dict):
        """Process entry signal with enhanced validation"""
        try:
            # Check anti-hedge logic
            if not self._check_anti_hedge(signal_type):
                self.logger.debug(f"Anti-hedge blocked {signal_type} signal")
                return
            
            # Enhanced signal strength validation
            signal_strength = signal_data.get("strength", 0)
            if signal_strength < 50:  # Minimum signal strength threshold
                self.logger.debug(f"Signal strength too low: {signal_strength}")
                return
            
            # Calculate position size
            base_volume = self.config.lot_size
            
            # Validate order size with risk manager
            valid, adjusted_volume, message = self.risk_manager.validate_order_size(
                base_volume, signal_type
            )
            
            if not valid:
                self.logger.warning(f"Order size validation failed: {message}")
                return
            
            # Calculate take profit
            tp_points = self.position_manager.calculate_take_profit(
                0 if signal_type == "BUY" else 1
            )
            
            # Execute order
            order_type = OrderType.MARKET_BUY if signal_type == "BUY" else OrderType.MARKET_SELL
            comment = f"{signal_type} Signal - RSI: {signal_data.get('rsi', 0):.1f} Str: {signal_strength:.1f}"
            
            result = self.order_executor.execute_market_order(
                order_type=order_type,
                volume=adjusted_volume,
                tp_points=tp_points,
                comment=comment
            )
            
            if result.success:
                self._on_trade_opened(result, signal_type, signal_data)
                # Record signal execution
                self.engine_stats['signals_generated'] += 1
            else:
                self.logger.error(f"Failed to execute {signal_type} order: {result.error_msg}")
                self._notify_error(f"Order execution failed: {result.error_msg}")
            
        except Exception as e:
            self.logger.error(f"Entry signal processing error: {e}")
            self._notify_error(f"Entry signal error: {e}")
    
    def _check_anti_hedge(self, signal_type: str) -> bool:
        """Enhanced anti-hedge logic"""
        current_positions = self.position_manager.positions
        
        for position in current_positions.values():
            # Basic direction check
            if ((position.type == 0 and signal_type == "SELL") or 
                (position.type == 1 and signal_type == "BUY")):
                return False
            
            # Check recovery positions
            if position.is_recovery:
                # Don't allow opposite signals during recovery
                if ((position.type == 0 and signal_type == "SELL") or 
                    (position.type == 1 and signal_type == "BUY")):
                    return False
        
        return True
    
    def _check_recovery_opportunities_smart(self):
        """Enhanced smart recovery with signal validation"""
        try:
            for ticket, position in self.position_manager.positions.items():
                if self.position_manager.check_recovery_needed(position):
                    
                    # Check if smart recovery is enabled
                    if self.config.smart_recovery:
                        # Wait for same signal type before recovery
                        signal_type = "BUY" if position.type == 0 else "SELL"
                        if not self._validate_recovery_signal(signal_type, position):
                            self.logger.debug(f"Smart recovery waiting for {signal_type} signal for position {ticket}")
                            continue
                    
                    # Find or create recovery group
                    group_id = None
                    for gid, group in self.position_manager.recovery_groups.items():
                        if position in group.positions:
                            group_id = gid
                            break
                    
                    if group_id is None:
                        group_id = self.position_manager.create_recovery_group(position)
                    
                    if self.position_manager.can_add_recovery(group_id):
                        recovery_result = self.order_executor.execute_recovery_order(
                            position, group_id
                        )
                        
                        if recovery_result.success:
                            self.engine_stats['recovery_triggered'] += 1
                            self.logger.info(f"✓ Recovery order executed for position {ticket}")
                        else:
                            self.logger.warning(f"✗ Recovery failed for position {ticket}: {recovery_result.error_msg}")
                            
        except Exception as e:
            self.logger.error(f"Recovery check error: {e}")
            self._notify_error(f"Recovery error: {e}")
    
    def _validate_recovery_signal(self, signal_type: str, position: Position) -> bool:
        """Validate if recovery signal is present for smart recovery"""
        try:
            # Check if we have recent signal for this type
            if signal_type not in self.recovery_signal_cache:
                return False
            
            cached_signal = self.recovery_signal_cache[signal_type]
            
            # Check signal age (should be recent)
            signal_age = (datetime.now() - cached_signal["timestamp"]).total_seconds()
            if signal_age > 300:  # 5 minutes max
                return False
            
            # Check signal strength
            if cached_signal["strength"] < 60:  # Higher threshold for recovery
                return False
            
            # Additional validation for recovery context
            rsi_value = cached_signal["rsi"]
            if signal_type == "BUY" and rsi_value < self.config.rsi_up:
                return False
            if signal_type == "SELL" and rsi_value > self.config.rsi_down:
                return False
            
            self.logger.info(f"✓ Smart recovery signal validated for {signal_type}: RSI={rsi_value:.1f}, Strength={cached_signal['strength']:.1f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Recovery signal validation error: {e}")
            return False
    
    def _monitor_positions(self):
        """Enhanced position monitoring with dynamic TP"""
        try:
            if self.config.dynamic_tp:
                for group_id, group in self.position_manager.recovery_groups.items():
                    if len(group.positions) > 1:
                        # Calculate new TP for recovery group
                        new_tp = self.position_manager.calculate_take_profit(
                            group.direction, group.positions
                        )
                        
                        # Update TP for all positions in group
                        for position in group.positions:
                            current_tp = position.tp
                            if abs(current_tp - new_tp) > 5:  # Only update if significant change
                                self.order_executor.modify_position_tp(position.ticket, new_tp)
                                self.logger.debug(f"Updated TP for {position.ticket}: {current_tp} → {new_tp}")
                            
        except Exception as e:
            self.logger.error(f"Position monitoring error: {e}")
    
    def _on_trade_opened(self, result: OrderResult, signal_type: str, signal_data: Dict):
        """Handle trade opened event with enhanced logging"""
        trade_info = {
            "ticket": result.ticket,
            "type": signal_type,
            "volume": result.volume,
            "price": result.price,
            "timestamp": datetime.now(),
            "signal_data": signal_data,
            "tp": result.tp if hasattr(result, 'tp') else 0,
            "sl": result.sl if hasattr(result, 'sl') else 0
        }
        
        self.trade_history.append(trade_info)
        self.engine_stats['trades_executed'] += 1
        
        # Queue event for handlers
        self.trade_event_queue.put(('trade_opened', trade_info))
        
        # Enhanced logging
        self.logger.info(f"📈 Trade opened: {signal_type} {result.volume} lots at {result.price}")
        self.logger.info(f"   Signal strength: {signal_data.get('strength', 0):.1f}, RSI: {signal_data.get('rsi', 0):.1f}")
    
    def _queue_status_update(self):
        """Queue comprehensive status update for UI"""
        try:
            status = {
                "timestamp": datetime.now(),
                "state": self.state.value,
                "connection_health": {
                    "connected": self.connection_health.is_connected,
                    "quality": self.connection_health.connection_quality,
                    "consecutive_failures": self.connection_health.consecutive_failures,
                    "response_time": self.connection_health.response_time_avg
                },
                "stats": self.engine_stats.copy(),
                "positions": {
                    "total": len(self.position_manager.positions),
                    "recovery_groups": len(self.position_manager.recovery_groups),
                    "total_pnl": sum(pos.profit for pos in self.position_manager.positions.values())
                },
                "signals": {
                    "cached_signals": len(self.recovery_signal_cache),
                    "last_signals": {k: v["timestamp"].isoformat() for k, v in self.last_successful_signals.items()}
                }
            }
            
            self.ui_update_queue.put(status)
            
        except Exception as e:
            self.logger.error(f"Status queue error: {e}")
    
    def _update_loop_timing(self, loop_time: float):
        """Update loop timing statistics"""
        if self.engine_stats['avg_loop_time'] == 0:
            self.engine_stats['avg_loop_time'] = loop_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.engine_stats['avg_loop_time'] = (
                alpha * loop_time + (1 - alpha) * self.engine_stats['avg_loop_time']
            )
        
        if loop_time > self.engine_stats['max_loop_time']:
            self.engine_stats['max_loop_time'] = loop_time
        
        # Log if loop is taking too long
        if loop_time > 1.0:
            self.logger.warning(f"Slow loop detected: {loop_time:.3f}s")
    
    def _validate_config(self) -> bool:
        """Enhanced configuration validation"""
        try:
            # Basic validations
            if self.config.lot_size <= 0:
                self.logger.error("Invalid lot size")
                return False
            
            if not (20 <= self.config.rsi_down <= 50):
                self.logger.error("Invalid RSI_DOWN range")
                return False
            
            if not (50 <= self.config.rsi_up <= 80):
                self.logger.error("Invalid RSI_UP range")
                return False
            
            if self.config.rsi_down >= self.config.rsi_up:
                self.logger.error("RSI_DOWN must be less than RSI_UP")
                return False
            
            # Recovery validations
            if self.config.martingale < 1.1 or self.config.martingale > 5.0:
                self.logger.error("Invalid martingale multiplier")
                return False
            
            if self.config.max_recovery < 1 or self.config.max_recovery > 10:
                self.logger.error("Invalid max recovery level")
                return False
            
            # Risk validations
            if self.config.daily_loss_limit <= 0:
                self.logger.error("Invalid daily loss limit")
                return False
            
            self.logger.info("✓ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def _update_engine_stats(self):
        """Update comprehensive engine statistics"""
        if self.start_time:
            self.engine_stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
        
        # Update positions count
        self.engine_stats['active_positions'] = len(self.position_manager.positions)
        self.engine_stats['recovery_groups'] = len(self.position_manager.recovery_groups)
        
        # Calculate success rate
        if self.engine_stats['trades_executed'] > 0:
            success_rate = (self.engine_stats['trades_executed'] - self.engine_stats['trades_closed']) / self.engine_stats['trades_executed']
            self.engine_stats['success_rate'] = success_rate * 100
    
    def _notify_state_change(self):
        """Thread-safe state change notification"""
        with self.event_handlers_lock:
            for handler in self.event_handlers['on_state_changed']:
                try:
                    handler(self.state)
                except Exception as e:
                    self.logger.error(f"State change handler error: {e}")
    
    def _notify_connection_status(self, status: str):
        """Notify connection status change"""
        with self.event_handlers_lock:
            for handler in self.event_handlers['on_connection_status']:
                try:
                    handler(status, self.connection_health)
                except Exception as e:
                    self.logger.error(f"Connection status handler error: {e}")
    
    def _notify_error(self, error_msg: str):
        """Thread-safe error notification"""
        self.error_queue.put(error_msg)
        
        with self.event_handlers_lock:
            for handler in self.event_handlers['on_error']:
                try:
                    handler(error_msg)
                except Exception as e:
                    self.logger.error(f"Error handler error: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Thread-safe event handler addition"""
        with self.event_handlers_lock:
            if event_type in self.event_handlers:
                self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Thread-safe event handler removal"""
        with self.event_handlers_lock:
            if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(handler)
    
    def get_status(self) -> EngineStatus:
        """Thread-safe status retrieval"""
        try:
            # Get risk report
            risk_report = self.risk_manager.get_risk_report()
            
            # Get position summary
            position_summary = self.position_manager.get_position_summary()
            
            # Get execution stats
            execution_stats = self.order_executor.get_execution_stats()
            
            uptime = self.engine_stats['uptime_seconds'] if self.start_time else 0
            
            return EngineStatus(
                state=self.state,
                uptime=uptime,
                last_update=self.last_update,
                total_trades=execution_stats.get('total_orders', 0),
                successful_trades=execution_stats.get('successful_orders', 0),
                current_positions=position_summary.get('total_positions', 0),
                total_pnl=position_summary.get('total_profit', 0),
                risk_level=RiskLevel(risk_report.get('risk_level', 'low')),
                restrictions=risk_report.get('restrictions', []),
                last_signal=self.signal_history[-1]['type'] if self.signal_history else "",
                last_trade=self.trade_history[-1]['timestamp'] if self.trade_history else None,
                errors=[f"Error count: {self.engine_stats['errors_occurred']}"],
                connection_status="connected" if self.connection_health.is_connected else "disconnected",
                reconnection_count=self.connection_health.total_reconnections
            )
            
        except Exception as e:
            self.logger.error(f"Get status error: {e}")
            return EngineStatus(
                state=EngineState.ERROR,
                errors=[f"Status error: {e}"]
            )
    
    def get_detailed_status(self) -> Dict:
        """Thread-safe detailed status retrieval"""
        try:
            status = self.get_status()
            
            return {
                "engine": {
                    "state": status.state.value,
                    "uptime": status.uptime,
                    "last_update": status.last_update.isoformat() if status.last_update else None,
                    "stats": self.engine_stats.copy(),
                    "connection": {
                        "connected": self.connection_health.is_connected,
                        "quality": self.connection_health.connection_quality,
                        "response_time": self.connection_health.response_time_avg,
                        "failures": self.connection_health.consecutive_failures,
                        "reconnections": self.connection_health.total_reconnections,
                        "last_error": self.connection_health.last_error
                    }
                },
                "trading": {
                    "total_trades": status.total_trades,
                    "successful_trades": status.successful_trades,
                    "current_positions": status.current_positions,
                    "total_pnl": status.total_pnl,
                    "last_trade": status.last_trade.isoformat() if status.last_trade else None
                },
                "signals": {
                    "cached_signals": len(self.recovery_signal_cache),
                    "last_signals": {k: v["timestamp"].isoformat() for k, v in self.last_successful_signals.items()},
                    "smart_recovery_enabled": self.config.smart_recovery
                },
                "risk": self.risk_manager.get_risk_report(),
                "positions": self.position_manager.get_position_summary(),
                "recovery": self.position_manager.get_recovery_status(),
                "execution": self.order_executor.get_execution_stats(),
                "config": self.config.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Detailed status error: {e}")
            return {"error": str(e)}
    
    # UI Helper Methods
    def get_ui_updates(self, timeout: float = 0.1) -> List[Dict]:
        """Get queued UI updates (non-blocking)"""
        updates = []
        try:
            while not self.ui_update_queue.empty():
                update = self.ui_update_queue.get(block=False)
                updates.append(update)
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Get UI updates error: {e}")
        
        return updates
    
    def get_trade_events(self, timeout: float = 0.1) -> List[Tuple]:
        """Get queued trade events (non-blocking)"""
        events = []
        try:
            while not self.trade_event_queue.empty():
                event = self.trade_event_queue.get(block=False)
                events.append(event)
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Get trade events error: {e}")
        
        return events
    
    def get_error_messages(self, timeout: float = 0.1) -> List[str]:
        """Get queued error messages (non-blocking)"""
        errors = []
        try:
            while not self.error_queue.empty():
                error = self.error_queue.get(block=False)
                errors.append(error)
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Get error messages error: {e}")
        
        return errors
    
    # Recovery Testing Methods
    def enable_recovery_test_mode(self, enable: bool = True):
        """Enable recovery testing mode"""
        self.recovery_test_mode = enable
        if enable:
            self.logger.info("🧪 Recovery test mode enabled")
        else:
            self.logger.info("Recovery test mode disabled")
    
    def test_recovery_system(self) -> Dict:
        """Enhanced recovery system testing"""
        try:
            self.logger.info("🧪 Testing recovery system...")
            test_results = {
                "timestamp": datetime.now(),
                "tests_passed": 0,
                "tests_failed": 0,
                "results": []
            }
            
            # Test 1: Recovery calculation
            try:
                base_volume = 0.01
                recovery_volume = self.position_manager.get_recovery_lot_size(base_volume, 1)
                expected = base_volume * self.config.martingale
                
                if abs(recovery_volume - expected) < 0.001:
                    test_results["tests_passed"] += 1
                    test_results["results"].append({"test": "Recovery volume calculation", "status": "PASS"})
                else:
                    test_results["tests_failed"] += 1
                    test_results["results"].append({
                        "test": "Recovery volume calculation", 
                        "status": "FAIL",
                        "expected": expected,
                        "actual": recovery_volume
                    })
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["results"].append({"test": "Recovery volume calculation", "status": "ERROR", "error": str(e)})
            
            # Test 2: TP calculation for recovery
            try:
                mock_positions = [
                    Position(
                        ticket=12345,
                        symbol=self.config.symbol,
                        type=0,  # BUY
                        volume=0.01,
                        open_price=2000.0,
                        open_time=datetime.now()
                    ),
                    Position(
                        ticket=12346,
                        symbol=self.config.symbol,
                        type=0,  # BUY
                        volume=0.02,
                        open_price=1995.0,
                        open_time=datetime.now()
                    )
                ]
                
                tp_points = self.position_manager.calculate_take_profit(0, mock_positions)
                
                if tp_points > 0:
                    test_results["tests_passed"] += 1
                    test_results["results"].append({"test": "Recovery TP calculation", "status": "PASS", "tp_points": tp_points})
                else:
                    test_results["tests_failed"] += 1
                    test_results["results"].append({"test": "Recovery TP calculation", "status": "FAIL", "tp_points": tp_points})
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["results"].append({"test": "Recovery TP calculation", "status": "ERROR", "error": str(e)})
            
            # Test 3: Anti-hedge logic
            try:
                can_buy = self._check_anti_hedge("BUY")
                can_sell = self._check_anti_hedge("SELL")
                
                if can_buy and can_sell:
                    test_results["tests_passed"] += 1
                    test_results["results"].append({"test": "Anti-hedge (no positions)", "status": "PASS"})
                else:
                    test_results["tests_failed"] += 1
                    test_results["results"].append({"test": "Anti-hedge (no positions)", "status": "FAIL"})
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["results"].append({"test": "Anti-hedge logic", "status": "ERROR", "error": str(e)})
            
            # Test 4: Smart recovery signal validation
            try:
                # Simulate cached signal
                self.recovery_signal_cache["BUY"] = {
                    "timestamp": datetime.now(),
                    "data": {"rsi": 58, "strength": 75},
                    "strength": 75,
                    "rsi": 58
                }
                
                mock_position = Position(
                    ticket=12345,
                    symbol=self.config.symbol,
                    type=0,  # BUY
                    volume=0.01,
                    open_price=2000.0,
                    open_time=datetime.now()
                )
                
                is_valid = self._validate_recovery_signal("BUY", mock_position)
                
                if is_valid:
                    test_results["tests_passed"] += 1
                    test_results["results"].append({"test": "Smart recovery signal validation", "status": "PASS"})
                else:
                    test_results["tests_failed"] += 1
                    test_results["results"].append({"test": "Smart recovery signal validation", "status": "FAIL"})
                    
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["results"].append({"test": "Smart recovery signal validation", "status": "ERROR", "error": str(e)})
            
            self.recovery_test_results.append(test_results)
            
            total_tests = test_results["tests_passed"] + test_results["tests_failed"]
            success_rate = (test_results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
            
            self.logger.info(f"🧪 Recovery test completed: {test_results['tests_passed']}/{total_tests} passed ({success_rate:.1f}%)")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Recovery test error: {e}")
            return {"error": str(e), "timestamp": datetime.now()}
    
    def save_state(self, filepath: str):
        """Enhanced save state with connection and signal info"""
        try:
            state_data = {
                "config": self.config.to_dict(),
                "stats": self.engine_stats.copy(),
                "connection_health": {
                    "total_reconnections": self.connection_health.total_reconnections,
                    "connection_quality": self.connection_health.connection_quality,
                    "response_time_avg": self.connection_health.response_time_avg
                },
                "trade_history": [
                    {**trade, "timestamp": trade["timestamp"].isoformat()} 
                    for trade in self.trade_history
                ],
                "signal_history": self.signal_history,
                "recovery_test_results": self.recovery_test_results,
                "cached_signals": {
                    k: {**v, "timestamp": v["timestamp"].isoformat()} 
                    for k, v in self.recovery_signal_cache.items()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.info(f"Engine state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self, filepath: str) -> bool:
        """Enhanced load state with signal cache"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore configuration
            self.config.update_from_dict(state_data.get("config", {}))
            
            # Restore statistics
            self.engine_stats.update(state_data.get("stats", {}))
            
            # Restore connection health history
            conn_health = state_data.get("connection_health", {})
            self.connection_health.total_reconnections = conn_health.get("total_reconnections", 0)
            
            # Restore histories
            self.signal_history = state_data.get("signal_history", [])
            self.recovery_test_results = state_data.get("recovery_test_results", [])
            
            # Restore cached signals
            cached_signals = state_data.get("cached_signals", {})
            for signal_type, signal_data in cached_signals.items():
                signal_data["timestamp"] = datetime.fromisoformat(signal_data["timestamp"])
                self.recovery_signal_cache[signal_type] = signal_data
            
            # Restore trade history with datetime conversion
            trade_history = state_data.get("trade_history", [])
            for trade in trade_history:
                if "timestamp" in trade:
                    trade["timestamp"] = datetime.fromisoformat(trade["timestamp"])
            self.trade_history = trade_history
            
            self.logger.info(f"Engine state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False