import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import queue
from concurrent.futures import ThreadPoolExecutor

# Fix circular import with TYPE_CHECKING
if TYPE_CHECKING:
    from strategy_engine import StrategyEngine, EngineState
    from trading_core import TradingConfig
    from risk_manager import RiskLevel
    from position_manager import Position

class UITheme(Enum):
    DARK = "dark"
    LIGHT = "light"
    CUSTOM = "custom"

@dataclass
class UIConfig:
    """UI Configuration"""
    theme: UITheme = UITheme.DARK
    update_interval: float = 1.0
    chart_history_bars: int = 100
    log_max_lines: int = 1000
    auto_scroll_logs: bool = True
    show_advanced_controls: bool = False
    position_in_title: bool = True
    sound_alerts: bool = True
    max_ui_updates_per_second: int = 10

class PresetManager:
    """Trading preset configurations"""
    
    PRESETS = {
        "Scalping": {
            "lot_size": 0.01,
            "rsi_up": 60,
            "rsi_down": 40,
            "tp_first": 150,
            "exit_speed": 0,  # FAST
            "recovery_price": 80,
            "martingale": 1.5,
            "max_recovery": 2,
            "primary_tf": "M5"
        },
        "Intraday": {
            "lot_size": 0.02,
            "rsi_up": 55,
            "rsi_down": 45,
            "tp_first": 200,
            "exit_speed": 1,  # MEDIUM
            "recovery_price": 100,
            "martingale": 2.0,
            "max_recovery": 3,
            "primary_tf": "M15"
        },
        "Swing": {
            "lot_size": 0.05,
            "rsi_up": 50,
            "rsi_down": 50,
            "tp_first": 300,
            "exit_speed": 2,  # SLOW
            "recovery_price": 150,
            "martingale": 2.5,
            "max_recovery": 4,
            "primary_tf": "H1"
        },
        "Conservative": {
            "lot_size": 0.01,
            "rsi_up": 65,
            "rsi_down": 35,
            "tp_first": 250,
            "exit_speed": 1,
            "recovery_price": 120,
            "martingale": 1.8,
            "max_recovery": 2,
            "primary_tf": "H1"
        }
    }

class ThreadSafeUIUpdater:
    """Thread-safe UI update manager"""
    
    def __init__(self, root: tk.Tk, max_updates_per_second: int = 10):
        self.root = root
        self.update_queue = queue.Queue(maxsize=100)
        self.is_updating = False
        self.last_update = 0
        self.min_update_interval = 1.0 / max_updates_per_second
        self.update_lock = threading.Lock()
        
    def schedule_update(self, update_func, *args, **kwargs):
        """Schedule a UI update function to run in main thread"""
        try:
            update_item = (update_func, args, kwargs)
            self.update_queue.put(update_item, block=False)
            
            # Schedule processing if not already scheduled
            with self.update_lock:
                if not self.is_updating:
                    self.is_updating = True
                    self.root.after_idle(self._process_updates)
                    
        except queue.Full:
            # Drop update if queue is full
            pass
    
    def _process_updates(self):
        """Process all queued updates"""
        try:
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_update < self.min_update_interval:
                # Reschedule for later
                self.root.after(
                    int((self.min_update_interval - (current_time - self.last_update)) * 1000),
                    self._process_updates
                )
                return
            
            updates_processed = 0
            max_updates_per_batch = 5  # Limit updates per batch
            
            while not self.update_queue.empty() and updates_processed < max_updates_per_batch:
                try:
                    update_func, args, kwargs = self.update_queue.get_nowait()
                    update_func(*args, **kwargs)
                    updates_processed += 1
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"UI update error: {e}")
            
            self.last_update = current_time
            
            # Schedule next batch if there are more updates
            if not self.update_queue.empty():
                self.root.after_idle(self._process_updates)
            else:
                with self.update_lock:
                    self.is_updating = False
                    
        except Exception as e:
            print(f"Update processing error: {e}")
            with self.update_lock:
                self.is_updating = False

class ConnectionStatusWidget:
    """Connection status display widget"""
    
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        
        # Connection indicator
        self.status_var = tk.StringVar(value="Disconnected")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        # Quality indicator
        self.quality_var = tk.StringVar(value="0%")
        ttk.Label(self.frame, text="Quality:").pack(side=tk.LEFT, padx=(10, 0))
        self.quality_label = ttk.Label(self.frame, textvariable=self.quality_var)
        self.quality_label.pack(side=tk.LEFT)
        
        # Reconnection count
        self.reconnect_var = tk.StringVar(value="0")
        ttk.Label(self.frame, text="Reconnects:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(self.frame, textvariable=self.reconnect_var).pack(side=tk.LEFT)
    
    def update_status(self, connected: bool, quality: float = 0, reconnections: int = 0):
        """Update connection status"""
        if connected:
            self.status_var.set("✅ Connected")
            color = "green"
        else:
            self.status_var.set("❌ Disconnected")
            color = "red"
        
        self.status_label.config(foreground=color)
        self.quality_var.set(f"{quality:.0f}%")
        self.reconnect_var.set(str(reconnections))

class XAUUSDTradingUI:
    def __init__(self):
        print("Initializing XAUUSD Trading UI with thread safety...")
        
        # Initialize basic state
        self.engine = None
        self.ui_config = UIConfig()
        self.preset_manager = PresetManager()
        
        # Thread safety
        self.ui_thread_id = threading.get_ident()
        self.ui_lock = threading.RLock()
        self.data_lock = threading.RLock()
        
        # UI state
        self.running = False
        self.update_thread = None
        self.last_update = None
        
        # Data for UI (thread-safe access)
        with self.data_lock:
            self.status_data = {}
            self.position_data = []
            self.recovery_data = []
            self.performance_data = {}
            self.connection_data = {}
        
        print("Creating main window...")
        # Create main window
        self.root = tk.Tk()
        self.root.title("XAUUSD Multi-Timeframe EA - Professional Trading System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize thread-safe updater
        self.ui_updater = ThreadSafeUIUpdater(self.root, self.ui_config.max_ui_updates_per_second)
        
        print("Setting up UI theme...")
        self.setup_styles()
        
        print("Creating UI components...")
        self.create_main_layout()
        self.create_control_panel()
        self.create_status_panel()
        self.create_trading_panel()
        self.create_risk_panel()
        self.create_positions_panel()
        self.create_logs_panel()
        
        print("Setting up logging...")
        self.setup_ui_logging()
        
        print("Setting up event bindings...")
        self.setup_event_bindings()
        
        # Initialize connection status widget
        self.connection_widget = ConnectionStatusWidget(self.root)
        
        print("Scheduling delayed engine initialization...")
        # Initialize engine after UI is ready
        self.root.after(500, self.delayed_engine_init)
        
        print("UI initialization complete!")
    
    def setup_styles(self):
        """Setup UI styles and themes"""
        self.style = ttk.Style()
        
        if self.ui_config.theme == UITheme.DARK:
            # Dark theme colors
            self.colors = {
                'bg': '#1e1e1e',
                'fg': '#ffffff',
                'select_bg': '#404040',
                'button_bg': '#404040',
                'success': '#00ff00',
                'warning': '#ffaa00',
                'error': '#ff4444',
                'profit': '#00aa00',
                'loss': '#aa0000'
            }
        else:
            # Light theme colors
            self.colors = {
                'bg': '#ffffff',
                'fg': '#000000',
                'select_bg': '#e0e0e0',
                'button_bg': '#f0f0f0',
                'success': '#008800',
                'warning': '#cc8800',
                'error': '#cc0000',
                'profit': '#006600',
                'loss': '#cc0000'
            }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg'])
        
        # Configure custom button styles
        try:
            self.style.configure("Success.TButton", background="#28a745")
            self.style.configure("Warning.TButton", background="#ffc107")
            self.style.configure("Danger.TButton", background="#dc3545")
        except:
            pass  # Fallback to default styles
    
    def delayed_engine_init(self):
        """Initialize engine after UI is fully loaded (thread-safe)"""
        self.logger.info("Starting delayed engine initialization...")
        
        # Use thread pool for engine initialization
        def init_engine():
            try:
                self.initialize_engine()
                self.ui_updater.schedule_update(self._on_engine_initialized)
            except Exception as e:
                self.logger.error(f"Engine initialization failed: {e}")
                self.ui_updater.schedule_update(
                    self._on_engine_init_failed, 
                    f"Failed to initialize trading engine: {e}"
                )
        
        # Run initialization in background thread
        threading.Thread(target=init_engine, daemon=True, name="EngineInit").start()
    
    def _on_engine_initialized(self):
        """Called when engine initialization succeeds (UI thread)"""
        self.logger.info("Engine initialization completed successfully")
        
        # Enable controls
        if hasattr(self, 'start_btn'):
            self.start_btn.config(state="normal")
        if hasattr(self, 'stop_btn'):
            self.stop_btn.config(state="normal")
        if hasattr(self, 'pause_btn'):
            self.pause_btn.config(state="normal")
        if hasattr(self, 'emergency_btn'):
            self.emergency_btn.config(state="normal")
    
    def _on_engine_init_failed(self, error_msg: str):
        """Called when engine initialization fails (UI thread)"""
        messagebox.showwarning("Engine Error", 
                             f"{error_msg}\n\nUI will continue in demo mode.")
    
    def create_main_layout(self):
        """Create main layout with panels"""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main trading tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Trading")
        
        # Advanced tab
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced")
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Configure main frame layout
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(1, weight=1)    
    
    def create_control_panel(self):
        """Create main control panel"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Engine Control", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Engine state display
        self.state_var = tk.StringVar(value="STOPPED")
        self.state_label = ttk.Label(control_frame, textvariable=self.state_var, 
                                   font=("Arial", 12, "bold"))
        self.state_label.grid(row=0, column=0, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=1, padx=20)
        
        self.start_btn = ttk.Button(button_frame, text="START", command=self.start_engine,
                                  state="disabled")
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="STOP", command=self.stop_engine,
                                 state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=2)
        
        self.pause_btn = ttk.Button(button_frame, text="PAUSE", command=self.pause_engine,
                                  state="disabled")
        self.pause_btn.grid(row=0, column=2, padx=2)
        
        self.emergency_btn = ttk.Button(button_frame, text="EMERGENCY STOP", 
                                      command=self.emergency_stop,
                                      state="disabled")
        self.emergency_btn.grid(row=0, column=3, padx=10)
        
        # Quick preset selector
        preset_frame = ttk.Frame(control_frame)
        preset_frame.grid(row=0, column=2, padx=20)
        
        ttk.Label(preset_frame, text="Quick Preset:").grid(row=0, column=0)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                  values=list(self.preset_manager.PRESETS.keys()),
                                  state="readonly", width=12)
        preset_combo.grid(row=0, column=1, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        # Connection status (enhanced)
        connection_frame = ttk.Frame(control_frame)
        connection_frame.grid(row=0, column=3, padx=20)
        
        self.connection_widget = ConnectionStatusWidget(connection_frame)
        self.connection_widget.frame.pack()
        
        # Uptime display
        uptime_frame = ttk.Frame(control_frame)
        uptime_frame.grid(row=0, column=4, padx=20)
        
        ttk.Label(uptime_frame, text="Uptime:").grid(row=0, column=0)
        self.uptime_var = tk.StringVar(value="00:00:00")
        ttk.Label(uptime_frame, textvariable=self.uptime_var).grid(row=0, column=1, padx=5)
        
        # Test and recovery buttons
        test_frame = ttk.Frame(control_frame)
        test_frame.grid(row=0, column=5, padx=20)
        
        self.test_btn = ttk.Button(test_frame, text="TEST LOG", command=self.test_log)
        self.test_btn.grid(row=0, column=0)
        
        self.test_recovery_btn = ttk.Button(test_frame, text="TEST RECOVERY", command=self.test_recovery)
        self.test_recovery_btn.grid(row=0, column=1, padx=5)
    
    def create_status_panel(self):
        """Create status and metrics panel"""
        status_frame = ttk.LabelFrame(self.main_frame, text="Live Status", padding="5")
        status_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Real-time metrics with enhanced display
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable text widget for status
        status_text_frame = ttk.Frame(metrics_frame)
        status_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_text_frame, height=15, width=40, 
                                 bg=self.colors['bg'], fg=self.colors['fg'],
                                 font=("Consolas", 9))
        status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", 
                                       command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Performance indicators
        perf_frame = ttk.LabelFrame(status_frame, text="Performance", padding="5")
        perf_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Loop timing display
        self.loop_time_var = tk.StringVar(value="Loop: 0.00ms")
        ttk.Label(perf_frame, textvariable=self.loop_time_var).pack(side=tk.LEFT)
        
        # Update rate display
        self.update_rate_var = tk.StringVar(value="Updates: 0/s")
        ttk.Label(perf_frame, textvariable=self.update_rate_var).pack(side=tk.LEFT, padx=(10, 0))
    
    def create_trading_panel(self):
        """Create trading parameters panel"""
        trading_frame = ttk.LabelFrame(self.main_frame, text="Trading Parameters", padding="5")
        trading_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create notebook for parameter categories
        param_notebook = ttk.Notebook(trading_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Entry Settings Tab
        entry_frame = ttk.Frame(param_notebook)
        param_notebook.add(entry_frame, text="Entry")
        self.create_entry_parameters(entry_frame)
        
        # Exit Settings Tab
        exit_frame = ttk.Frame(param_notebook)
        param_notebook.add(exit_frame, text="Exit")
        self.create_exit_parameters(exit_frame)
        
        # Recovery Settings Tab
        recovery_frame = ttk.Frame(param_notebook)
        param_notebook.add(recovery_frame, text="Recovery")
        self.create_recovery_parameters(recovery_frame)
        
        # Risk Settings Tab
        risk_frame = ttk.Frame(param_notebook)
        param_notebook.add(risk_frame, text="Risk")
        self.create_risk_parameters(risk_frame)
    
    def create_entry_parameters(self, parent):
        """Create entry parameter controls"""
        # Lot Size
        row = 0
        ttk.Label(parent, text="Lot Size:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.lot_size_var = tk.DoubleVar(value=0.01)
        lot_spin = ttk.Spinbox(parent, from_=0.01, to=10.0, increment=0.01, 
                              textvariable=self.lot_size_var, width=10)
        lot_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # RSI Upper
        row += 1
        ttk.Label(parent, text="RSI Upper:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.rsi_up_var = tk.IntVar(value=55)
        rsi_up_spin = ttk.Spinbox(parent, from_=50, to=80, increment=1, 
                                 textvariable=self.rsi_up_var, width=10)
        rsi_up_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # RSI Lower
        row += 1
        ttk.Label(parent, text="RSI Lower:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.rsi_down_var = tk.IntVar(value=45)
        rsi_down_spin = ttk.Spinbox(parent, from_=20, to=50, increment=1, 
                                   textvariable=self.rsi_down_var, width=10)
        rsi_down_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Trading Direction
        row += 1
        ttk.Label(parent, text="Direction:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.direction_var = tk.StringVar(value="BOTH")
        direction_combo = ttk.Combobox(parent, textvariable=self.direction_var,
                                     values=["BOTH", "BUY_ONLY", "SELL_ONLY", "STOP"],
                                     state="readonly", width=12)
        direction_combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Primary Timeframe
        row += 1
        ttk.Label(parent, text="Timeframe:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.timeframe_var = tk.StringVar(value="M15")
        tf_combo = ttk.Combobox(parent, textvariable=self.timeframe_var,
                               values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                               state="readonly", width=12)
        tf_combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Apply button
        row += 1
        apply_btn = ttk.Button(parent, text="Apply Changes", command=self.apply_parameters)
        apply_btn.grid(row=row, column=0, columnspan=2, pady=10)
    
    def create_exit_parameters(self, parent):
        """Create exit parameter controls"""
        # Take Profit
        row = 0
        ttk.Label(parent, text="TP Points:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.tp_first_var = tk.IntVar(value=200)
        tp_spin = ttk.Spinbox(parent, from_=50, to=1000, increment=10, 
                             textvariable=self.tp_first_var, width=10)
        tp_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Exit Speed
        row += 1
        ttk.Label(parent, text="Exit Speed:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.exit_speed_var = tk.StringVar(value="MEDIUM")
        speed_combo = ttk.Combobox(parent, textvariable=self.exit_speed_var,
                                  values=["FAST", "MEDIUM", "SLOW"],
                                  state="readonly", width=12)
        speed_combo.grid(row=row, column=1, padx=5, pady=2)
        
        # Dynamic TP
        row += 1
        self.dynamic_tp_var = tk.BooleanVar(value=True)
        dynamic_check = ttk.Checkbutton(parent, text="Dynamic TP for Recovery",
                                       variable=self.dynamic_tp_var)
        dynamic_check.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
    
    def create_recovery_parameters(self, parent):
        """Create recovery parameter controls"""
        # Recovery Price
        row = 0
        ttk.Label(parent, text="Recovery at Loss:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.recovery_price_var = tk.IntVar(value=100)
        recovery_spin = ttk.Spinbox(parent, from_=50, to=500, increment=10, 
                                   textvariable=self.recovery_price_var, width=10)
        recovery_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="points").grid(row=row, column=2, sticky="w", padx=5)
        
        # Martingale
        row += 1
        ttk.Label(parent, text="Martingale:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.martingale_var = tk.DoubleVar(value=2.0)
        martingale_spin = ttk.Spinbox(parent, from_=1.1, to=5.0, increment=0.1, 
                                     textvariable=self.martingale_var, width=10)
        martingale_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Max Recovery
        row += 1
        ttk.Label(parent, text="Max Recovery:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_recovery_var = tk.IntVar(value=3)
        max_recovery_spin = ttk.Spinbox(parent, from_=1, to=10, increment=1, 
                                       textvariable=self.max_recovery_var, width=10)
        max_recovery_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Smart Recovery
        row += 1
        self.smart_recovery_var = tk.BooleanVar(value=True)
        smart_check = ttk.Checkbutton(parent, text="Smart Recovery (Wait for same signal)",
                                     variable=self.smart_recovery_var)
        smart_check.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
    
    def create_risk_parameters(self, parent):
        """Create risk parameter controls"""
        # Daily Loss Limit
        row = 0
        ttk.Label(parent, text="Daily Loss Limit:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.daily_loss_var = tk.DoubleVar(value=100.0)
        daily_spin = ttk.Spinbox(parent, from_=10, to=1000, increment=10, 
                                textvariable=self.daily_loss_var, width=10)
        daily_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="USD").grid(row=row, column=2, sticky="w", padx=5)
        
        # Max Drawdown
        row += 1
        ttk.Label(parent, text="Max Drawdown:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_drawdown_var = tk.DoubleVar(value=10.0)
        drawdown_spin = ttk.Spinbox(parent, from_=1, to=50, increment=1, 
                                   textvariable=self.max_drawdown_var, width=10)
        drawdown_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="%").grid(row=row, column=2, sticky="w", padx=5)
        
        # Max Positions
        row += 1
        ttk.Label(parent, text="Max Positions:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_positions_var = tk.IntVar(value=5)
        positions_spin = ttk.Spinbox(parent, from_=1, to=20, increment=1, 
                                    textvariable=self.max_positions_var, width=10)
        positions_spin.grid(row=row, column=1, padx=5, pady=2)
        
        # Max Spread
        row += 1
        ttk.Label(parent, text="Max Spread:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.max_spread_var = tk.IntVar(value=30)
        spread_spin = ttk.Spinbox(parent, from_=5, to=100, increment=5, 
                                 textvariable=self.max_spread_var, width=10)
        spread_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="points").grid(row=row, column=2, sticky="w", padx=5)
        
        # Min Account Balance
        row += 1
        ttk.Label(parent, text="Min Balance:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.min_balance_var = tk.DoubleVar(value=500.0)
        balance_spin = ttk.Spinbox(parent, from_=100, to=10000, increment=100, 
                                  textvariable=self.min_balance_var, width=10)
        balance_spin.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(parent, text="USD").grid(row=row, column=2, sticky="w", padx=5)
    
    def create_risk_panel(self):
        """Create risk monitoring panel"""
        risk_frame = ttk.LabelFrame(self.advanced_frame, text="Risk Monitor", padding="5")
        risk_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Risk level indicator
        level_frame = ttk.Frame(risk_frame)
        level_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(level_frame, text="Risk Level:").pack(side=tk.LEFT)
        self.risk_level_var = tk.StringVar(value="LOW")
        self.risk_level_label = ttk.Label(level_frame, textvariable=self.risk_level_var,
                                         font=("Arial", 12, "bold"))
        self.risk_level_label.pack(side=tk.LEFT, padx=10)
        
        # Risk metrics
        metrics_frame = ttk.Frame(risk_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.risk_text = tk.Text(metrics_frame, height=20, 
                               bg=self.colors['bg'], fg=self.colors['fg'],
                               font=("Consolas", 9))
        risk_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", 
                                     command=self.risk_text.yview)
        self.risk_text.configure(yscrollcommand=risk_scrollbar.set)
        
        self.risk_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        risk_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_positions_panel(self):
        """Create positions monitoring panel"""
        positions_frame = ttk.LabelFrame(self.analysis_frame, text="Active Positions", padding="5")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Position tree view
        columns = ("Ticket", "Type", "Volume", "Open Price", "Current Price", "Profit", "Recovery")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient="vertical", 
                                          command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Position control buttons
        button_frame = ttk.Frame(positions_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Close Selected", 
                  command=self.close_selected_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close All", 
                  command=self.close_all_positions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh", 
                  command=self.refresh_positions).pack(side=tk.LEFT, padx=5)
    
    def create_logs_panel(self):
        """Create logging panel"""
        logs_frame = ttk.LabelFrame(self.analysis_frame, text="System Logs", padding="5")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log text widget
        self.log_text = tk.Text(logs_frame, height=15, 
                              bg=self.colors['bg'], fg=self.colors['fg'],
                              font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", 
                                    command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(log_controls, text="Clear", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Export", command=self.export_logs).pack(side=tk.LEFT, padx=5)
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto Scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=10)
    
    def create_risk_panel(self):
        """Create risk monitoring panel"""
        risk_frame = ttk.LabelFrame(self.advanced_frame, text="Risk Monitor", padding="5")
        risk_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Risk level indicator
        level_frame = ttk.Frame(risk_frame)
        level_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(level_frame, text="Risk Level:").pack(side=tk.LEFT)
        self.risk_level_var = tk.StringVar(value="🟢 LOW")
        self.risk_level_label = ttk.Label(level_frame, textvariable=self.risk_level_var,
                                        font=("Arial", 12, "bold"))
        self.risk_level_label.pack(side=tk.LEFT, padx=10)
        
        # Risk metrics text area
        metrics_frame = ttk.Frame(risk_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.risk_text = tk.Text(metrics_frame, height=25, 
                            bg=self.colors['bg'], fg=self.colors['fg'],
                            font=("Consolas", 9), wrap=tk.NONE)
        risk_scrollbar_v = ttk.Scrollbar(metrics_frame, orient="vertical", 
                                    command=self.risk_text.yview)
        risk_scrollbar_h = ttk.Scrollbar(metrics_frame, orient="horizontal",
                                    command=self.risk_text.xview)
        self.risk_text.configure(yscrollcommand=risk_scrollbar_v.set,
                            xscrollcommand=risk_scrollbar_h.set)
        
        self.risk_text.grid(row=0, column=0, sticky="nsew")
        risk_scrollbar_v.grid(row=0, column=1, sticky="ns")
        risk_scrollbar_h.grid(row=1, column=0, sticky="ew")
        
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)
        
        # Control buttons
        control_frame = ttk.Frame(risk_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Refresh Risk Data", 
                command=self.refresh_risk_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Risk Report", 
                command=self.export_risk_report).pack(side=tk.LEFT, padx=5)
        
        # Initialize with demo data
        self._update_risk_display_demo()

    def _update_risk_display_demo(self):
        """Show demo risk data when engine not available"""
        try:
            risk_info = f"""
                                      RISK MONITOR - DEMO MODE                            
                                                                                          
     📊 ACCOUNT STATUS                                                                    
       Balance:              $1,000.00                                                   
       Equity:               $1,000.00                                                   
       Free Margin:          $1,000.00                                                   
       Margin Level:         999.99%                                                     
       Used Margin:          0.00%                                                       
       Account Type:         DEMO                                                        
       Currency:             USD                                                         
       Leverage:             1:100                                                       
                                                                                          
     📈 P&L TRACKING                                                                     
       Daily P&L:            $0.00                                                       
       Weekly P&L:           $0.00                                                       
       Monthly P&L:          $0.00                                                       
       Daily Target:         $20.00                                                      
       Weekly Target:        $100.00                                                     
                                                                                          
     📉 DRAWDOWN ANALYSIS                                                                
       Current Drawdown:     0.00%                                                       
       Max Drawdown:         0.00%                                                       
       Peak Balance:         $1,000.00                                                   
       Peak Equity:          $1,000.00                                                   
       Balance Drawdown:     0.00%                                                       
       Equity Drawdown:      0.00%                                                       
                                                                                          
     🎯 PERFORMANCE METRICS                                                              
       Total Trades:         0                                                           
       Winning Trades:       0                                                           
       Losing Trades:        0                                                           
       Win Rate:             0.00%                                                       
       Profit Factor:        0.00                                                        
       Average Win:          $0.00                                                       
       Average Loss:         $0.00                                                       
                                                                                          
     ⚙️ RISK LIMITS & THRESHOLDS                                                         
       Daily Loss Limit:     $100.00                                                     
       Weekly Loss Limit:    $500.00                                                     
       Monthly Loss Limit:   $2,000.00                                                   
       Max Drawdown:         10.00%                                                      
       Max Positions:        5                                                           
       Min Margin Level:     200.00%                                                     
       Max Used Margin:      50.00%                                                      
       Emergency Stop:       100.00%                                                     
                                                                                          
     🌐 MARKET CONDITIONS                                                                
       Market Session:       Asian                                                       
       Volatility:           Low (0.5%)                                                  
       Trend Strength:       Neutral (0.0)                                              
       Current Spread:       30 points                                                   
       Market Status:        Open                                                        
       Trading Allowed:      Yes                                                         
       High Impact News:     No                                                          
                                                                                          
     📊 POSITION ANALYSIS                                                                
       Active Positions:     0                                                           
       Total Volume:         0.00 lots                                                   
       Recovery Groups:      0                                                           
       Position Exposure:    0.00%                                                       
       Largest Position:     $0.00                                                       
       Position Correlation: 0.00%                                                       
                                                                                          
     ⚠️ CURRENT RESTRICTIONS                                                             
       🟢 No restrictions active                                                         
       🟢 Trading fully allowed                                                          
       🟢 All risk parameters within limits                                              
       🟢 Connection stable                                                              
       🟢 Market conditions favorable                                                    
                                                                                          
     🔧 SYSTEM STATUS                                                                    
       Engine State:         RUNNING                                                     
       Last Update:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}               
       Data Source:          Demo/Default Values                                         
       Update Frequency:     5 seconds                                                   
       Risk Calculation:     Active                                                      
                                                                                          

    💡 RISK LEVEL EXPLANATION:
    🟢 LOW     - All parameters within safe limits
    🟡 MEDIUM  - Some parameters approaching limits  
    🟠 HIGH    - Risk parameters exceeded, careful monitoring
    🔴 CRITICAL- Emergency conditions, trading may be restricted

    📋 NEXT ACTIONS:
    • Monitor account balance and equity changes
    • Watch for position correlation and exposure
    • Keep daily loss within configured limits
    • Maintain margin level above minimum threshold
            """
            
            self.risk_text.delete('1.0', tk.END)
            self.risk_text.insert('1.0', risk_info)
            
        except Exception as e:
            self.logger.error(f"Risk display demo error: {e}")
            self.risk_text.delete('1.0', tk.END)
            self.risk_text.insert('1.0', f"Error displaying risk data: {e}")

    def refresh_risk_data(self):
            """Refresh risk data manually"""
            try:
                if hasattr(self, 'engine') and self.engine:
                    self.logger.info("Refreshing risk data...")
                    risk_data = self.engine.risk_manager.get_risk_report()
                    self.update_risk_display_real(risk_data)
                else:
                    self._update_risk_display_demo()
                    self.logger.info("Engine not available, showing demo data")
            except Exception as e:
                self.logger.error(f"Refresh risk data error: {e}")

    def update_risk_display_real(self, risk_data: Dict):
        """Update risk display with real engine data"""
        try:
            if 'error' in risk_data:
                self.risk_text.delete('1.0', tk.END)
                self.risk_text.insert('1.0', f"Error retrieving risk data: {risk_data['error']}")
                return

            # Update risk level indicator
            risk_level = risk_data.get('risk_level', 'low').upper()
            risk_icons = {
                'LOW': '🟢',
                'MEDIUM': '🟡', 
                'HIGH': '🟠',
                'CRITICAL': '🔴'
            }
            risk_icon = risk_icons.get(risk_level, '🟢')
            self.risk_level_var.set(f"{risk_icon} {risk_level}")
            
            # Update risk level label color
            colors = {
                'LOW': 'green',
                'MEDIUM': 'orange',
                'HIGH': 'red', 
                'CRITICAL': 'red'
            }
            self.risk_level_label.config(foreground=colors.get(risk_level, 'green'))

            # Format comprehensive risk display
            account = risk_data.get('account', {})
            pnl = risk_data.get('pnl', {})
            drawdown = risk_data.get('drawdown', {})
            performance = risk_data.get('performance', {})
            market = risk_data.get('market', {})
            limits = risk_data.get('limits', {})
            
            # Get trading status
            trading_allowed = risk_data.get('trading_allowed', False)
            restrictions = risk_data.get('restrictions', [])
            
            risk_info = f"""
                                  LIVE RISK MONITOR                                   

                                                                                      
 📊 ACCOUNT STATUS                                                                     
   Balance:              ${account.get('balance', 0):,.2f}                             
   Equity:               ${account.get('equity', 0):,.2f}                              
   Free Margin:          ${account.get('free_margin', 0):,.2f}                         
   Margin Level:         {account.get('margin_level', 0):,.2f}%                           
   Used Margin:          {account.get('used_margin_percent', 0):.2f}%                
   Account Type:         {'LIVE' if not risk_data.get('Real', True) else 'Real'}     
   Currency:             {account.get('currency', 'USD')}                            
   Leverage:             1:{account.get('leverage', 0)}                              
                                                                                      
 📈 P&L TRACKING                                                                     
   Daily P&L:            ${pnl.get('daily_pnl', 0):+,.2f}                           
   Weekly P&L:           ${pnl.get('weekly_pnl', 0):+,.2f}                          
   Monthly P&L:          ${pnl.get('monthly_pnl', 0):+,.2f}                         
   Daily Target:         ${limits.get('daily_loss_limit', 0):,.2f}                  
   Daily Remaining:      ${limits.get('daily_loss_limit', 0) - abs(pnl.get('daily_pnl', 0)):,.2f}  
                                                                                      
 📉 DRAWDOWN ANALYSIS                                                                
   Current Drawdown:     {drawdown.get('current_drawdown', 0):.2f}%                 
   Max Drawdown:         {drawdown.get('max_drawdown', 0):.2f}%                     
   Peak Balance:         ${drawdown.get('peak_balance', 0):,.2f}                    
   Peak Equity:          ${drawdown.get('peak_equity', 0):,.2f}                     
   Balance Drawdown:     {drawdown.get('balance_drawdown', 0):.2f}%                 
   Equity Drawdown:      {drawdown.get('equity_drawdown', 0):.2f}%                  
                                                                                      
 🎯 PERFORMANCE METRICS                                                              
   Total Trades:         {performance.get('total_trades', 0)}                       
   Winning Trades:       {performance.get('winning_trades', 0)}                     
   Losing Trades:        {performance.get('losing_trades', 0)}                      
   Win Rate:             {performance.get('win_rate', 0):.2f}%                      
   Profit Factor:        {performance.get('profit_factor', 0):.2f}                  
   Average Win:          ${performance.get('avg_win', 0):+,.2f}                     
   Average Loss:         ${performance.get('avg_loss', 0):+,.2f}                    
                                                                                      
 ⚙️ RISK LIMITS & THRESHOLDS                                                         
   Daily Loss Limit:     ${limits.get('daily_loss_limit', 0):,.2f}                 
   Max Drawdown:         {limits.get('max_drawdown_percent', 0):.2f}%               
   Max Positions:        {limits.get('max_positions', 0)}                          
   Min Margin Level:     {limits.get('min_margin_level', 0):,.2f}%                  
   Max Used Margin:      {limits.get('max_used_margin', 0):.2f}%                    
                                                                                      
 🌐 MARKET CONDITIONS                                                                
   Market Session:       {market.get('session', 'Unknown').title()}                 
   Volatility:           {market.get('volatility', 0):.1f}%                         
   Trend Strength:       {market.get('trend_strength', 0):.1f}                      
   Current Spread:       {market.get('current_spread', 0)} points                   
   High Spread Alert:    {'YES' if market.get('high_spread', False) else 'NO'}      
   Low Liquidity:        {'YES' if market.get('low_liquidity', False) else 'NO'}    
   Market Closed:        {'YES' if market.get('market_closed', False) else 'NO'}    
                                                                                      
 📊 POSITION ANALYSIS                                                                """

            # Add position data if available
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'position_manager'):
                pos_summary = self.engine.position_manager.get_position_summary()
                risk_info += f"""
   Active Positions:     {pos_summary.get('total_positions', 0)}                    
   Total Volume:         {pos_summary.get('total_volume', 0):.2f} lots              
   Recovery Groups:      {pos_summary.get('recovery_groups', 0)}                    
   Buy Positions:        {pos_summary.get('buy_positions', 0)}                     
   Sell Positions:       {pos_summary.get('sell_positions', 0)}                    
   Total P&L:            ${pos_summary.get('total_profit', 0):+,.2f}               """
            else:
                risk_info += f"""
   Active Positions:     0                                                           
   Total Volume:         0.00 lots                                                   
   Recovery Groups:      0                                                           
   Position Exposure:    0.00%                                                       """

            # Trading restrictions section
            risk_info += f"""
                                                                                      
 ⚠️ TRADING STATUS & RESTRICTIONS                                                    
   Trading Allowed:      {'🟢 YES' if trading_allowed else '🔴 NO'}                 """

            if restrictions:
                risk_info += f"""
   Active Restrictions:  {len(restrictions)} restriction(s)                         """
                for i, restriction in enumerate(restrictions[:3]):  # Show first 3 restrictions
                    risk_info += f"""
   • {restriction[:65]:<65} """
                if len(restrictions) > 3:
                    risk_info += f"""
   • ... and {len(restrictions) - 3} more restrictions                              """
            else:
                risk_info += f"""
   Active Restrictions:  🟢 None - All systems operational                          """

            # Risk level explanation
            risk_info += f"""
                                                                                      
 🔧 SYSTEM STATUS                                                                    
   Risk Level:           {risk_icon} {risk_level}                                   
   Data Valid:           {'🟢 YES' if risk_data.get('data_valid', False) else '🔴 NO'} 
   Last Update:          {risk_data.get('last_update', 'Never')[:19] if risk_data.get('last_update') else 'Never'} 
   Update Source:        Live MT5 Data                                              
   Monitoring Active:    🟢 YES                                                     
                                                                                      

💡 RISK LEVEL EXPLANATION:
🟢 LOW     - All parameters within safe limits, normal trading
🟡 MEDIUM  - Some parameters approaching limits, monitor closely  
🟠 HIGH    - Risk parameters exceeded, careful monitoring required
🔴 CRITICAL- Emergency conditions, trading may be restricted/stopped

📋 CURRENT STATUS:
• Account health: {'Good' if account.get('margin_level', 0) > 200 else 'Warning' if account.get('margin_level', 0) > 100 else 'Critical'}
• Daily P&L status: {'Positive' if pnl.get('daily_pnl', 0) >= 0 else 'Negative'}
• Position exposure: {'Normal' if drawdown.get('current_drawdown', 0) < 5 else 'Elevated'}
• Market conditions: {'Favorable' if not market.get('high_spread', False) else 'Challenging'}

⚡ NEXT ACTIONS:
• {'Continue monitoring' if trading_allowed else 'Address restrictions before trading'}
• {'Manage open positions carefully' if pos_summary.get('total_positions', 0) > 0 else 'Ready for new signals'}
• {'Monitor daily loss closely' if abs(pnl.get('daily_pnl', 0)) > limits.get('daily_loss_limit', 100) * 0.5 else 'Daily loss within limits'}
            """
            
            # Clear and update display
            self.risk_text.delete('1.0', tk.END)
            self.risk_text.insert('1.0', risk_info)
            
        except Exception as e:
            self.logger.error(f"Real risk display update error: {e}")
            # Fallback to demo display
            self._update_risk_display_demo()

    def export_risk_report(self):
        """Export risk report to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Risk Report"
            )
            
            if filename:
                risk_content = self.risk_text.get('1.0', tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(risk_content)
                
                messagebox.showinfo("Success", f"Risk report exported to {filename}")
                
        except Exception as e:
            self.logger.error(f"Export risk report error: {e}")
            messagebox.showerror("Error", f"Failed to export risk report: {e}")


    def setup_event_bindings(self):
        """Setup event bindings"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind parameter changes
        self.lot_size_var.trace_add("write", self.on_parameter_changed)
        self.rsi_up_var.trace_add("write", self.on_parameter_changed)
        self.rsi_down_var.trace_add("write", self.on_parameter_changed)
    
    def setup_ui_logging(self):
        """Setup UI logging handler with thread safety"""
        self.logger = logging.getLogger("XAUUSD_EA")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        class ThreadSafeUILogHandler(logging.Handler):
            def __init__(self, ui_instance):
                super().__init__()
                self.ui = ui_instance
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Use UI updater for thread-safe logging
                    self.ui.ui_updater.schedule_update(self.ui._add_log, msg)
                except Exception as e:
                    print(f"Log handler error: {e}")
        
        # Add UI log handler
        ui_handler = ThreadSafeUILogHandler(self)
        ui_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', 
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(ui_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        self.logger.info("UI logging system initialized")
    
    def _add_log(self, msg: str):
        """Add log message to UI (must be called from UI thread)"""
        try:
            if threading.get_ident() != self.ui_thread_id:
                self.logger.warning("_add_log called from non-UI thread")
                return
            
            self.log_text.insert(tk.END, f"{msg}\n")
            
            # Auto scroll if enabled
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)
            
            # Limit log lines
            lines = int(self.log_text.index('end-1c').split('.')[0])
            if lines > self.ui_config.log_max_lines:
                self.log_text.delete('1.0', f'{lines - self.ui_config.log_max_lines}.0')
                
        except Exception as e:
            print(f"Add log error: {e}")
    
    def initialize_engine(self):
        """Initialize trading engine with enhanced error handling"""
        self.logger.info("Importing trading modules...")
        
        try:
            # Import modules
            from trading_core import TradingConfig
            from strategy_engine import StrategyEngine
            self.logger.info("✓ Core modules imported successfully")
            
            # Create configuration
            config = TradingConfig()
            
            # Update config with current UI values
            self._update_config_from_ui(config)
            
            self.logger.info("✓ Trading configuration created")
            
            # Create engine
            self.engine = StrategyEngine(config)
            self.logger.info("✓ Strategy engine created")
            
            # Setup event handlers
            self._setup_engine_event_handlers()
            self.logger.info("✓ Event handlers configured")
            
            # Test MT5 connection
            self.logger.info("Testing MT5 connection...")
            if self.engine.trading_core.initialize_mt5():
                self.logger.info("✓ MT5 connection successful")
                self.ui_updater.schedule_update(self._update_connection_status, True, 100, 0)
            else:
                self.logger.warning("✗ MT5 connection failed")
                self.ui_updater.schedule_update(self._update_connection_status, False, 0, 0)
            
            self.logger.info("🎉 Engine initialization completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Engine initialization failed: {e}")
            raise
    
    def _update_config_from_ui(self, config):
        """Update config with current UI values"""
        config.lot_size = self.lot_size_var.get()
        config.rsi_up = self.rsi_up_var.get()
        config.rsi_down = self.rsi_down_var.get()
        config.tp_first = self.tp_first_var.get()
        config.recovery_price = self.recovery_price_var.get()
        config.martingale = self.martingale_var.get()
        config.max_recovery = self.max_recovery_var.get()
        config.daily_loss_limit = self.daily_loss_var.get()
        config.max_drawdown = self.max_drawdown_var.get()
        config.max_positions = self.max_positions_var.get()
        config.max_spread_alert = self.max_spread_var.get()
        config.min_account_balance = self.min_balance_var.get()
        config.primary_tf = self.timeframe_var.get()
        config.dynamic_tp = self.dynamic_tp_var.get()
        config.smart_recovery = self.smart_recovery_var.get()
        
        # Map direction
        direction_map = {"BOTH": 0, "BUY_ONLY": 1, "SELL_ONLY": 2, "STOP": 3}
        config.trading_direction = direction_map.get(self.direction_var.get(), 0)
        
        # Map exit speed
        exit_speed_map = {"FAST": 0, "MEDIUM": 1, "SLOW": 2}
        config.exit_speed = exit_speed_map.get(self.exit_speed_var.get(), 1)
    
    def _setup_engine_event_handlers(self):
        """Setup engine event handlers"""
        self.engine.add_event_handler('on_trade_opened', self.on_trade_opened)
        self.engine.add_event_handler('on_trade_closed', self.on_trade_closed)
        self.engine.add_event_handler('on_state_changed', self.on_engine_state_changed)
        self.engine.add_event_handler('on_error', self.on_engine_error)
        self.engine.add_event_handler('on_connection_status', self.on_connection_status_changed)
    
    def _update_connection_status(self, connected: bool, quality: float, reconnections: int):
        """Update connection status display (UI thread)"""
        if hasattr(self, 'connection_widget'):
            self.connection_widget.update_status(connected, quality, reconnections)
    
    def start_ui_updates(self):
        """Start UI update thread"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.ui_update_loop, daemon=True, name="UI_Updates")
            self.update_thread.start()
            self.logger.info("UI update thread started")
    
    def stop_ui_updates(self):
        """Stop UI update thread"""
        self.running = False
        if self.update_thread:
            self.logger.info("UI update thread stopped")
    
    def ui_update_loop(self):
        """Enhanced UI update loop"""
        self.logger.info("UI update loop started")
        last_update_time = time.time()
        update_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Get updates from engine
                if self.engine:
                    self._process_engine_updates()
                
                # Update performance metrics
                update_count += 1
                if current_time - last_update_time >= 1.0:
                    update_rate = update_count / (current_time - last_update_time)
                    self.ui_updater.schedule_update(self._update_performance_display, update_rate)
                    
                    update_count = 0
                    last_update_time = current_time
                
                time.sleep(self.ui_config.update_interval)
                
            except Exception as e:
                self.logger.error(f"UI update loop error: {e}")
                time.sleep(1.0)
    
    def _process_engine_updates(self):
        """Process updates from engine"""
        try:
            # Get UI updates
            ui_updates = self.engine.get_ui_updates()
            for update in ui_updates:
                self.ui_updater.schedule_update(self._handle_ui_update, update)
            
            # Get trade events
            trade_events = self.engine.get_trade_events()
            for event_type, event_data in trade_events:
                self.ui_updater.schedule_update(self._handle_trade_event, event_type, event_data)
            
            # Get error messages
            error_messages = self.engine.get_error_messages()
            for error_msg in error_messages:
                self.ui_updater.schedule_update(self._handle_error_message, error_msg)
                
        except Exception as e:
            self.logger.error(f"Process engine updates error: {e}")
    
    def _handle_ui_update(self, update: Dict):
        """Handle UI update with debug info"""
        try:
            
            # ทดสอบใส่ข้อมูลใน status_text
            status_info = f"""
    ENGINE STATE: {update.get('state', 'UNKNOWN')}
    TIMESTAMP: {datetime.now().strftime('%H:%M:%S')}
    CONNECTION: {'Connected' if update.get('connection_health', {}).get('connected', False) else 'Disconnected'}
    STATS: {update.get('stats', {})}
            """
            
            # Force update status text
            self.status_text.delete('1.0', tk.END)
            self.status_text.insert('1.0', status_info)
            
            # Original update logic...
            if 'state' in update:
                self.state_var.set(update['state'].upper())
            
        except Exception as e:
            print(f"UI Update Error: {e}")
            self.status_text.delete('1.0', tk.END)
            self.status_text.insert('1.0', f"ERROR: {e}")

    def _handle_trade_event(self, event_type: str, event_data: Dict):
        """Handle trade event (UI thread)"""
        try:
            if event_type == 'trade_opened':
                self.logger.info(f"📈 Trade Event: {event_data}")
            elif event_type == 'trade_closed':
                self.logger.info(f"📉 Trade Event: {event_data}")
                
        except Exception as e:
            self.logger.error(f"Handle trade event error: {e}")
    
    def _handle_error_message(self, error_msg: str):
        """Handle error message (UI thread)"""
        self.logger.error(f"Engine Error: {error_msg}")
    
    def _update_performance_display(self, update_rate: float):
        """Update performance display (UI thread)"""
        self.update_rate_var.set(f"Updates: {update_rate:.1f}/s")
    
    # Event handlers
    def start_engine(self):
        """Start trading engine (thread-safe)"""
        def start_engine_async():
            try:
                if self.engine:
                    self.logger.info("Starting trading engine...")
                    if self.engine.start():
                        self.ui_updater.schedule_update(self._on_engine_started)
                        self.start_ui_updates()
                    else:
                        self.ui_updater.schedule_update(self._on_engine_start_failed)
                else:
                    self.ui_updater.schedule_update(
                        lambda: messagebox.showwarning("No Engine", "Engine not initialized")
                    )
            except Exception as e:
                self.logger.error(f"Start engine error: {e}")
                self.ui_updater.schedule_update(
                    lambda: messagebox.showerror("Error", f"Failed to start engine: {e}")
                )
        
        # Run in background thread
        threading.Thread(target=start_engine_async, daemon=True, name="StartEngine").start()
    
    def _on_engine_started(self):
        """Called when engine starts successfully (UI thread)"""
        self.logger.info("✓ Engine started successfully")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
    
    def _on_engine_start_failed(self):
        """Called when engine start fails (UI thread)"""
        self.logger.error("✗ Failed to start engine")
        messagebox.showerror("Error", "Failed to start trading engine")
    
    def stop_engine(self):
        """Stop trading engine (thread-safe)"""
        def stop_engine_async():
            try:
                if self.engine:
                    self.logger.info("Stopping trading engine...")
                    self.engine.stop()
                    self.stop_ui_updates()
                    self.ui_updater.schedule_update(self._on_engine_stopped)
            except Exception as e:
                self.logger.error(f"Stop engine error: {e}")
        
        # Run in background thread
        threading.Thread(target=stop_engine_async, daemon=True, name="StopEngine").start()
    
    def _on_engine_stopped(self):
        """Called when engine stops (UI thread)"""
        self.logger.info("✓ Engine stopped")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def pause_engine(self):
        """Pause trading engine"""
        try:
            if self.engine:
                self.engine.pause()
                self.logger.info("Engine paused")
        except Exception as e:
            self.logger.error(f"Pause engine error: {e}")
    
    def emergency_stop(self):
        """Emergency stop"""
        if messagebox.askyesno("Emergency Stop", 
                              "This will close all positions and stop trading. Continue?"):
            def emergency_stop_async():
                try:
                    if self.engine:
                        self.engine.emergency_stop()
                        self.stop_ui_updates()
                        self.ui_updater.schedule_update(lambda: self.logger.critical("🚨 EMERGENCY STOP EXECUTED"))
                    else:
                        self.ui_updater.schedule_update(lambda: self.logger.info("Demo mode - emergency stop simulated"))
                except Exception as e:
                    self.logger.error(f"Emergency stop error: {e}")
            
            # Run in background thread
            threading.Thread(target=emergency_stop_async, daemon=True, name="EmergencyStop").start()
    
    def test_recovery(self):
        """Test recovery system"""
        def test_recovery_async():
            try:
                if self.engine:
                    self.logger.info("🧪 Running recovery system test...")
                    results = self.engine.test_recovery_system()
                    
                    self.ui_updater.schedule_update(self._show_recovery_test_results, results)
                else:
                    self.ui_updater.schedule_update(
                        lambda: self.logger.info("No engine - recovery test skipped")
                    )
            except Exception as e:
                self.logger.error(f"Recovery test error: {e}")
        
        # Run in background thread
        threading.Thread(target=test_recovery_async, daemon=True, name="RecoveryTest").start()
    
    def _show_recovery_test_results(self, results: Dict):
        """Show recovery test results (UI thread)"""
        if 'error' in results:
            messagebox.showerror("Test Error", f"Recovery test failed: {results['error']}")
            return
        
        passed = results.get('tests_passed', 0)
        failed = results.get('tests_failed', 0)
        total = passed + failed
        
        if total > 0:
            success_rate = (passed / total) * 100
            message = f"Recovery Test Results:\n\n"
            message += f"Tests Passed: {passed}\n"
            message += f"Tests Failed: {failed}\n"
            message += f"Success Rate: {success_rate:.1f}%\n\n"
            
            for result in results.get('results', []):
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                message += f"{status_icon} {result['test']}: {result['status']}\n"
            
            messagebox.showinfo("Recovery Test", message)
        else:
            messagebox.showwarning("Recovery Test", "No tests were executed")
    
    def apply_parameters(self):
        """Apply parameter changes to engine (thread-safe)"""
        def apply_params_async():
            try:
                if not self.engine:
                    self.ui_updater.schedule_update(lambda: self.logger.info("No engine - parameters saved locally"))
                    return
                
                # Validate RSI values
                rsi_up = self.rsi_up_var.get()
                rsi_down = self.rsi_down_var.get()
                
                if rsi_down >= rsi_up:
                    self.ui_updater.schedule_update(
                        lambda: messagebox.showerror("Parameter Error", 
                                       f"RSI Lower ({rsi_down}) must be less than RSI Upper ({rsi_up})")
                    )
                    return
                
                # Collect parameters
                params = self._collect_parameters()
                
                # Apply to engine
                self.engine.update_config(params)
                self.ui_updater.schedule_update(lambda: self.logger.info("✓ Parameters applied successfully"))
                
            except Exception as e:
                self.logger.error(f"Apply parameters error: {e}")
                self.ui_updater.schedule_update(
                    lambda: messagebox.showerror("Error", f"Failed to apply parameters: {e}")
                )
        
        # Run in background thread
        threading.Thread(target=apply_params_async, daemon=True, name="ApplyParams").start()
    
    def _collect_parameters(self) -> Dict:
        """Collect parameters from UI"""
        # Map enums
        direction_map = {"BOTH": 0, "BUY_ONLY": 1, "SELL_ONLY": 2, "STOP": 3}
        exit_speed_map = {"FAST": 0, "MEDIUM": 1, "SLOW": 2}
        
        return {
            "lot_size": self.lot_size_var.get(),
            "rsi_up": self.rsi_up_var.get(),
            "rsi_down": self.rsi_down_var.get(),
            "tp_first": self.tp_first_var.get(),
            "recovery_price": self.recovery_price_var.get(),
            "martingale": self.martingale_var.get(),
            "max_recovery": self.max_recovery_var.get(),
            "daily_loss_limit": self.daily_loss_var.get(),
            "max_drawdown": self.max_drawdown_var.get(),
            "max_positions": self.max_positions_var.get(),
            "max_spread_alert": self.max_spread_var.get(),
            "min_account_balance": self.min_balance_var.get(),
            "primary_tf": self.timeframe_var.get(),
            "dynamic_tp": self.dynamic_tp_var.get(),
            "smart_recovery": self.smart_recovery_var.get(),
            "trading_direction": direction_map.get(self.direction_var.get(), 0),
            "exit_speed": exit_speed_map.get(self.exit_speed_var.get(), 1)
        }
    
    def test_log(self):
        """Test logging system"""
        self.logger.info("=== UI TEST ===")
        self.logger.info(f"Current time: {datetime.now()}")
        self.logger.info(f"Engine status: {'Connected' if self.engine else 'Not connected'}")
        self.logger.info(f"UI thread: {threading.get_ident() == self.ui_thread_id}")
        self.logger.warning("Test warning message")
        self.logger.error("Test error message")
        self.logger.info("=== END TEST ===")
    
    def on_parameter_changed(self, *args):
        """Handle parameter change"""
        # Disabled auto-apply to prevent spam
        pass
    
    def on_preset_selected(self, event=None):
        """Handle preset selection"""
        preset_name = self.preset_var.get()
        if preset_name:
            self.load_preset(preset_name)
    
    def load_preset(self, preset_name):
        """Load trading preset"""
        try:
            if preset_name not in self.preset_manager.PRESETS:
                return
            
            preset = self.preset_manager.PRESETS[preset_name]
            
            # Update UI variables
            self.lot_size_var.set(preset["lot_size"])
            self.rsi_up_var.set(preset["rsi_up"])
            self.rsi_down_var.set(preset["rsi_down"])
            self.tp_first_var.set(preset["tp_first"])
            self.recovery_price_var.set(preset["recovery_price"])
            self.martingale_var.set(preset["martingale"])
            self.max_recovery_var.set(preset["max_recovery"])
            self.timeframe_var.set(preset["primary_tf"])
            
            # Map exit speed
            speed_names = ["FAST", "MEDIUM", "SLOW"]
            self.exit_speed_var.set(speed_names[preset["exit_speed"]])
            
            self.logger.info(f"✓ Loaded preset: {preset_name}")
            
        except Exception as e:
            self.logger.error(f"Load preset error: {e}")
            messagebox.showerror("Error", f"Failed to load preset: {e}")
    
    # Engine event handlers
    def on_trade_opened(self, trade_info):
        """Handle trade opened event"""
        self.logger.info(f"📈 Trade opened: {trade_info}")
    
    def on_trade_closed(self, trade_info):
        """Handle trade closed event"""
        self.logger.info(f"📉 Trade closed: {trade_info}")
    
    def on_engine_state_changed(self, new_state):
        """Handle engine state change"""
        try:
            state_str = new_state.value if hasattr(new_state, 'value') else str(new_state)
            self.ui_updater.schedule_update(lambda: self.state_var.set(state_str.upper()))
            self.logger.info(f"🔄 Engine state: {state_str}")
        except Exception as e:
            self.logger.error(f"State change error: {e}")
    
    def on_engine_error(self, error_msg):
        """Handle engine error"""
        self.logger.error(f"🚨 Engine error: {error_msg}")
    
    def on_connection_status_changed(self, status: str, connection_health):
        """Handle connection status change"""
        self.logger.info(f"🔗 Connection status: {status}")
        
        # Update connection widget
        self.ui_updater.schedule_update(
            self._update_connection_status,
            connection_health.is_connected,
            connection_health.connection_quality,
            connection_health.total_reconnections
        )
    
    # Utility methods
    def clear_logs(self):
        """Clear log display"""
        try:
            self.log_text.delete('1.0', tk.END)
        except:
            pass
    
    def export_logs(self):
        """Export logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                logs = self.log_text.get('1.0', tk.END)
                with open(filename, 'w') as f:
                    f.write(logs)
                
                messagebox.showinfo("Success", "Logs exported successfully")
                
        except Exception as e:
            self.logger.error(f"Export logs error: {e}")
            messagebox.showerror("Error", f"Failed to export logs: {e}")
    
    def close_selected_position(self):
        """Close selected position"""
        self.logger.info("Close selected position requested")
    
    def close_all_positions(self):
        """Close all positions"""
        if messagebox.askyesno("Confirm", "Close all positions?"):
            def close_all_async():
                try:
                    if self.engine and hasattr(self.engine, 'order_executor'):
                        self.logger.info("Closing all positions...")
                        # This should be implemented in engine
                    else:
                        self.ui_updater.schedule_update(lambda: self.logger.info("No active positions to close"))
                except Exception as e:
                    self.logger.error(f"Close all positions error: {e}")
            
            # Run in background thread
            threading.Thread(target=close_all_async, daemon=True, name="CloseAll").start()
    
    def refresh_positions(self):
        """Refresh positions display"""
        def refresh_async():
            try:
                if self.engine and hasattr(self.engine, 'position_manager'):
                    self.engine.position_manager.update_positions()
                    self.ui_updater.schedule_update(lambda: self.logger.info("Positions refreshed"))
                else:
                    self.ui_updater.schedule_update(lambda: self.logger.info("No position manager available"))
            except Exception as e:
                self.logger.error(f"Refresh positions error: {e}")
        
        # Run in background thread
        threading.Thread(target=refresh_async, daemon=True, name="RefreshPositions").start()
    
    def on_closing(self):
        """Handle window closing (thread-safe)"""
        try:
            self.logger.info("Shutting down application...")
            self.stop_ui_updates()
            
            # Check if engine is running
            if self.engine:
                try:
                    from strategy_engine import EngineState
                    if hasattr(self.engine, 'state') and self.engine.state not in [EngineState.STOPPED, EngineState.ERROR]:
                        if messagebox.askyesno("Confirm Exit", 
                                              "Trading engine is running. Stop and exit?"):
                            # Stop engine in background
                            def stop_and_exit():
                                try:
                                    self.engine.stop()
                                    self.ui_updater.schedule_update(self.root.destroy)
                                except:
                                    self.ui_updater.schedule_update(self.root.destroy)
                            
                            threading.Thread(target=stop_and_exit, daemon=True).start()
                            return  # Don't destroy immediately
                        else:
                            return  # Cancel exit
                except ImportError:
                    pass
            
            self.root.destroy()
            
        except Exception as e:
            print(f"Shutdown error: {e}")
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        try:
            self.logger.info("🚀 XAUUSD Trading UI Started")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
        finally:
            try:
                if self.engine and hasattr(self.engine, 'stop'):
                    self.engine.stop()
            except:
                pass
            self.logger.info("Application terminated")

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("XAUUSD Multi-Timeframe EA - Professional Trading System")
    print("=" * 60)
    
    try:
        app = XAUUSDTradingUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        input("Press Enter to exit...")