"""
XAUUSD Trading UI - Merged from ui_controller.py and advanced_ui.py
Professional Trading Interface
"""

# Standard imports
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from collections import deque
import sqlite3
import os
from pathlib import Path

# Internal imports - fixed for new structure
from src.core.trading_engine import StrategyEngine, TradingConfig

# ============================================================================
# ADVANCED UI COMPONENTS (from advanced_ui.py)
# ============================================================================


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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from collections import deque
import sqlite3
import os
from pathlib import Path

# Fix circular import with TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.trading_engine import StrategyEngine, EngineState
    from src.core.trading_engine import TradingConfig
    from src.core.risk_manager import RiskLevel
    from src.core.position_manager import Position

class AlertType(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"
    SIGNAL = "signal"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    auto_dismiss: bool = True
    dismiss_after: int = 5000  # milliseconds
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }

class DataStore:
    """SQLite data store for historical data"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    metadata TEXT
                )
            ''')
            
            # Trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    ticket INTEGER,
                    symbol TEXT,
                    type TEXT,
                    volume REAL,
                    open_price REAL,
                    close_price REAL,
                    profit REAL,
                    duration_seconds INTEGER
                )
            ''')
            
            # Signal history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    signal_type TEXT,
                    strength REAL,
                    confidence REAL,
                    rsi_value REAL,
                    executed BOOLEAN,
                    result TEXT
                )
            ''')
            
            # Account snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    balance REAL,
                    equity REAL,
                    margin REAL,
                    free_margin REAL,
                    margin_level REAL,
                    total_positions INTEGER
                )
            ''')
            
            conn.commit()
    
    def store_performance_metric(self, metric_name: str, value: float, unit: str = "", metadata: Dict = None):
        """Store performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, metric_name, value, unit, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), metric_name, value, unit, json.dumps(metadata or {})))
    
    def store_trade(self, trade_data: Dict):
        """Store trade data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trade_history (timestamp, ticket, symbol, type, volume, open_price, close_price, profit, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                trade_data.get('ticket', 0),
                trade_data.get('symbol', ''),
                trade_data.get('type', ''),
                trade_data.get('volume', 0),
                trade_data.get('open_price', 0),
                trade_data.get('close_price', 0),
                trade_data.get('profit', 0),
                trade_data.get('duration_seconds', 0)
            ))
    
    def store_signal(self, signal_data: Dict):
        """Store signal data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signal_history (timestamp, signal_type, strength, confidence, rsi_value, executed, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                signal_data.get('signal_type', ''),
                signal_data.get('strength', 0),
                signal_data.get('confidence', 0),
                signal_data.get('rsi_value', 0),
                signal_data.get('executed', False),
                signal_data.get('result', '')
            ))
    
    def store_account_snapshot(self, account_data: Dict):
        """Store account snapshot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO account_snapshots (timestamp, balance, equity, margin, free_margin, margin_level, total_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                account_data.get('balance', 0),
                account_data.get('equity', 0),
                account_data.get('margin', 0),
                account_data.get('free_margin', 0),
                account_data.get('margin_level', 0),
                account_data.get('total_positions', 0)
            ))
    
    def get_performance_data(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get performance data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, value, unit, metadata FROM performance_metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (metric_name, cutoff_time))
            
            return [
                {
                    "timestamp": row[0],
                    "value": row[1],
                    "unit": row[2],
                    "metadata": json.loads(row[3])
                }
                for row in cursor.fetchall()
            ]
    
    def get_account_history(self, hours: int = 24) -> List[Dict]:
        """Get account history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, balance, equity, margin_level FROM account_snapshots
                WHERE timestamp > ?
                ORDER BY timestamp
            ''', (cutoff_time,))
            
            return [
                {
                    "timestamp": row[0],
                    "balance": row[1],
                    "equity": row[2],
                    "margin_level": row[3]
                }
                for row in cursor.fetchall()
            ]

class AlertManager:
    """Alert management system"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.alerts = {}
        self.alert_counter = 0
        self.max_alerts = 10
        
        # Alert display frame
        self.alerts_frame = ttk.Frame(parent_widget)
        self.alerts_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Sound alerts (optional)
        self.sound_enabled = True
        
    def add_alert(self, alert_type: AlertType, title: str, message: str, auto_dismiss: bool = True):
        """Add new alert"""
        alert_id = f"alert_{self.alert_counter}"
        self.alert_counter += 1
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            auto_dismiss=auto_dismiss
        )
        
        # Remove oldest alerts if at max capacity
        if len(self.alerts) >= self.max_alerts:
            oldest_id = min(self.alerts.keys(), key=lambda x: self.alerts[x].timestamp)
            self.remove_alert(oldest_id)
        
        self.alerts[alert_id] = alert
        self._display_alert(alert)
        
        # Auto-dismiss if configured
        if auto_dismiss:
            self.parent.after(alert.dismiss_after, lambda: self.remove_alert(alert_id))
        
        # Play sound if enabled
        if self.sound_enabled and alert_type in [AlertType.ERROR, AlertType.WARNING, AlertType.TRADE]:
            self._play_alert_sound(alert_type)
    
    def _display_alert(self, alert: Alert):
        """Display alert in UI"""
        # Create alert frame
        alert_frame = ttk.Frame(self.alerts_frame, relief="raised", borderwidth=1)
        alert_frame.pack(fill=tk.X, pady=1)
        
        # Color coding
        colors = {
            AlertType.INFO: "#E3F2FD",
            AlertType.SUCCESS: "#E8F5E8", 
            AlertType.WARNING: "#FFF3E0",
            AlertType.ERROR: "#FFEBEE",
            AlertType.TRADE: "#E0F2F1",
            AlertType.SIGNAL: "#F3E5F5"
        }
        
        alert_frame.configure(style=f"{alert.type.value.title()}.TFrame")
        
        # Alert content
        content_frame = ttk.Frame(alert_frame)
        content_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Icon and title
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill=tk.X)
        
        # Icon
        icons = {
            AlertType.INFO: "ℹ️",
            AlertType.SUCCESS: "✅",
            AlertType.WARNING: "⚠️", 
            AlertType.ERROR: "❌",
            AlertType.TRADE: "💰",
            AlertType.SIGNAL: "📊"
        }
        
        icon_label = ttk.Label(header_frame, text=icons.get(alert.type, "•"), font=("Arial", 12))
        icon_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Title
        title_label = ttk.Label(header_frame, text=alert.title, font=("Arial", 10, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Timestamp
        time_label = ttk.Label(header_frame, text=alert.timestamp.strftime("%H:%M:%S"), 
                              font=("Arial", 8), foreground="gray")
        time_label.pack(side=tk.RIGHT)
        
        # Message
        if alert.message:
            message_label = ttk.Label(content_frame, text=alert.message, font=("Arial", 9))
            message_label.pack(fill=tk.X, pady=(2, 0))
        
        # Close button
        close_btn = ttk.Button(content_frame, text="×", width=3,
                              command=lambda: self.remove_alert(alert.id))
        close_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Store reference to frame for removal
        alert.frame = alert_frame
    
    def remove_alert(self, alert_id: str):
        """Remove alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            if hasattr(alert, 'frame'):
                alert.frame.destroy()
            del self.alerts[alert_id]
    
    def clear_all_alerts(self):
        """Clear all alerts"""
        for alert_id in list(self.alerts.keys()):
            self.remove_alert(alert_id)
    
    def _play_alert_sound(self, alert_type: AlertType):
        """Play alert sound (placeholder)"""
        # This would play system sounds or custom audio files
        # For now, just use system bell
        try:
            if alert_type in [AlertType.ERROR, AlertType.WARNING]:
                self.parent.bell()
        except Exception as e:
            print(f"Alert sound error: {e}")

class LiveChartWidget:
    """Live chart widget for real-time data visualization"""
    
    def __init__(self, parent, title: str = "Live Chart", max_points: int = 100):
        self.parent = parent
        self.title = title
        self.max_points = max_points
        
        # Data storage
        self.data_series = {}
        self.timestamps = deque(maxlen=max_points)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
        
        # Animation
        self.animation = None
        self.is_animating = False
        
        # Colors for different series
        self.colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        self.color_index = 0
    
    def add_data_series(self, name: str, color: str = None):
        """Add a new data series"""
        if name not in self.data_series:
            if color is None:
                color = self.colors[self.color_index % len(self.colors)]
                self.color_index += 1
            
            self.data_series[name] = {
                'data': deque(maxlen=self.max_points),
                'color': color,
                'line': None
            }
    
    def update_data(self, series_name: str, value: float, timestamp: datetime = None):
        """Update data for a series"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add series if it doesn't exist
        if series_name not in self.data_series:
            self.add_data_series(series_name)
        
        # Add data point
        self.data_series[series_name]['data'].append(value)
        
        # Update timestamps (only need one timeline)
        if len(self.timestamps) == 0 or timestamp > self.timestamps[-1]:
            self.timestamps.append(timestamp)
    
    def start_animation(self, interval: int = 1000):
        """Start real-time animation"""
        if not self.is_animating:
            self.animation = FuncAnimation(
                self.fig, self._animate, interval=interval, blit=False
            )
            self.is_animating = True
            self.canvas.draw()
    
    def stop_animation(self):
        """Stop animation"""
        if self.animation:
            self.animation.event_source.stop()
            self.is_animating = False
    
    def _animate(self, frame):
        """Animation function"""
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3)
        
        if len(self.timestamps) < 2:
            return
        
        # Plot each data series
        for name, series in self.data_series.items():
            if len(series['data']) > 0:
                # Ensure data and timestamps have same length
                data_len = len(series['data'])
                time_data = list(self.timestamps)[-data_len:]
                
                self.ax.plot(time_data, list(series['data']), 
                           label=name, color=series['color'], linewidth=2)
        
        # Format x-axis for timestamps
        self.ax.tick_params(axis='x', rotation=45)
        
        # Legend
        if self.data_series:
            self.ax.legend()
        
        # Adjust layout
        self.fig.tight_layout()
    
    def manual_refresh(self):
        """Manually refresh the chart"""
        self._animate(None)
        self.canvas.draw()

class PerformanceDashboard:
    """Performance dashboard with multiple charts and metrics"""
    
    def __init__(self, parent):
        self.parent = parent
        self.data_store = DataStore()
        
        # Create dashboard frame
        self.dashboard_frame = ttk.Frame(parent)
        self.dashboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different chart types
        self.chart_notebook = ttk.Notebook(self.dashboard_frame)
        self.chart_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Account Performance Tab
        self.account_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.account_frame, text="Account")
        self.account_chart = LiveChartWidget(self.account_frame, "Account Performance")
        self.account_chart.add_data_series("Balance", "blue")
        self.account_chart.add_data_series("Equity", "green")
        
        # Signal Analysis Tab
        self.signals_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.signals_frame, text="Signals")
        self.signals_chart = LiveChartWidget(self.signals_frame, "Signal Strength")
        self.signals_chart.add_data_series("Signal Strength", "purple")
        self.signals_chart.add_data_series("Confidence", "orange")
        
        # Performance Metrics Tab
        self.performance_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.performance_frame, text="Performance")
        self.performance_chart = LiveChartWidget(self.performance_frame, "System Performance")
        self.performance_chart.add_data_series("Loop Time (ms)", "red")
        self.performance_chart.add_data_series("Memory Usage", "brown")
        
        # Risk Metrics Tab
        self.risk_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.risk_frame, text="Risk")
        self.risk_chart = LiveChartWidget(self.risk_frame, "Risk Metrics")
        self.risk_chart.add_data_series("Drawdown %", "red")
        self.risk_chart.add_data_series("Margin Level", "blue")
        
        # Control panel
        self.create_control_panel()
        
        # Auto-refresh timer
        self.auto_refresh = True
        self.refresh_interval = 5000  # 5 seconds
        self.start_auto_refresh()
    
    def create_control_panel(self):
        """Create dashboard control panel"""
        control_frame = ttk.Frame(self.dashboard_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto Refresh", 
                       variable=self.auto_refresh_var,
                       command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        # Manual refresh button
        ttk.Button(control_frame, text="Refresh Now", 
                  command=self.manual_refresh).pack(side=tk.LEFT, padx=5)
        
        # Export button
        ttk.Button(control_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Time range selector
        ttk.Label(control_frame, text="Time Range:").pack(side=tk.LEFT, padx=(20, 5))
        self.time_range_var = tk.StringVar(value="1h")
        time_combo = ttk.Combobox(control_frame, textvariable=self.time_range_var,
                                values=["15m", "30m", "1h", "4h", "1d", "1w"],
                                state="readonly", width=8)
        time_combo.pack(side=tk.LEFT, padx=5)
        time_combo.bind("<<ComboboxSelected>>", self.on_time_range_changed)
        
        # Chart type selector
        ttk.Label(control_frame, text="Chart:").pack(side=tk.LEFT, padx=(20, 5))
        chart_types = ["Account", "Signals", "Performance", "Risk"]
        for chart_type in chart_types:
            ttk.Button(control_frame, text=chart_type, width=10,
                      command=lambda ct=chart_type: self.switch_chart(ct)).pack(side=tk.LEFT, padx=2)
    
    def update_account_data(self, balance: float, equity: float):
        """Update account performance chart"""
        timestamp = datetime.now()
        self.account_chart.update_data("Balance", balance, timestamp)
        self.account_chart.update_data("Equity", equity, timestamp)
        
        # Store in database
        self.data_store.store_account_snapshot({
            "balance": balance,
            "equity": equity,
            "margin": 0,
            "free_margin": 0,
            "margin_level": 999.99,
            "total_positions": 0
        })
    
    def update_signal_data(self, strength: float, confidence: float):
        """Update signal analysis chart"""
        timestamp = datetime.now()
        self.signals_chart.update_data("Signal Strength", strength, timestamp)
        self.signals_chart.update_data("Confidence", confidence, timestamp)
    
    def update_performance_data(self, loop_time_ms: float, memory_usage: float = 0):
        """Update performance metrics chart"""
        timestamp = datetime.now()
        self.performance_chart.update_data("Loop Time (ms)", loop_time_ms, timestamp)
        if memory_usage > 0:
            self.performance_chart.update_data("Memory Usage", memory_usage, timestamp)
        
        # Store in database
        self.data_store.store_performance_metric("loop_time", loop_time_ms, "ms")
    
    def update_risk_data(self, drawdown_percent: float, margin_level: float):
        """Update risk metrics chart"""
        timestamp = datetime.now()
        self.risk_chart.update_data("Drawdown %", drawdown_percent, timestamp)
        self.risk_chart.update_data("Margin Level", margin_level, timestamp)
    
    def start_auto_refresh(self):
        """Start auto-refresh timer"""
        if self.auto_refresh:
            self.refresh_charts()
            self.parent.after(self.refresh_interval, self.start_auto_refresh)
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh"""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self.start_auto_refresh()
    
    def manual_refresh(self):
        """Manually refresh all charts"""
        self.refresh_charts()
    
    def refresh_charts(self):
        """Refresh all charts"""
        try:
            self.account_chart.manual_refresh()
            self.signals_chart.manual_refresh()
            self.performance_chart.manual_refresh()
            self.risk_chart.manual_refresh()
        except Exception as e:
            print(f"Chart refresh error: {e}")
    
    def switch_chart(self, chart_type: str):
        """Switch to specific chart tab"""
        chart_map = {
            "Account": 0,
            "Signals": 1,
            "Performance": 2,
            "Risk": 3
        }
        
        if chart_type in chart_map:
            self.chart_notebook.select(chart_map[chart_type])
    
    def on_time_range_changed(self, event=None):
        """Handle time range change"""
        # This would reload data for the selected time range
        # For now, just refresh the charts
        self.manual_refresh()
    
    def export_data(self):
        """Export chart data to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                # Export current chart data
                current_tab = self.chart_notebook.select()
                tab_index = self.chart_notebook.index(current_tab)
                
                chart_map = {
                    0: self.account_chart,
                    1: self.signals_chart,
                    2: self.performance_chart,
                    3: self.risk_chart
                }
                
                chart = chart_map.get(tab_index)
                if chart:
                    self._export_chart_data(chart, filename)
                    messagebox.showinfo("Success", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _export_chart_data(self, chart: LiveChartWidget, filename: str):
        """Export chart data to CSV file"""
        data_rows = []
        
        # Get timestamps
        timestamps = list(chart.timestamps)
        
        # Get all series data
        for name, series in chart.data_series.items():
            data = list(series['data'])
            
            # Align data with timestamps
            for i, (timestamp, value) in enumerate(zip(timestamps[-len(data):], data)):
                if len(data_rows) <= i:
                    data_rows.append({"timestamp": timestamp})
                data_rows[i][name] = value
        
        # Write to CSV
        if data_rows:
            import csv
            
            fieldnames = ["timestamp"] + list(chart.data_series.keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data_rows)

class ConfigurationManager:
    """Advanced configuration management"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Current configuration
        self.current_config = {}
        
        # Configuration history
        self.config_history = []
        self.max_history = 50
    
    def save_config(self, config: Dict, name: str = None, description: str = ""):
        """Save configuration with metadata"""
        if name is None:
            name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_data = {
            "name": name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "version": "1.0"
        }
        
        # Save to file
        config_file = self.config_dir / f"{name}.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Add to history
        self.config_history.append(config_data)
        if len(self.config_history) > self.max_history:
            self.config_history.pop(0)
        
        return str(config_file)
    
    def load_config(self, name: str) -> Dict:
        """Load configuration by name"""
        config_file = self.config_dir / f"{name}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return config_data.get("config", {})
        
        return {}
    
    def list_configs(self) -> List[Dict]:
        """List all saved configurations"""
        configs = []
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    configs.append({
                        "name": config_data.get("name", config_file.stem),
                        "description": config_data.get("description", ""),
                        "timestamp": config_data.get("timestamp", ""),
                        "file": str(config_file)
                    })
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")
                continue
        
        return sorted(configs, key=lambda x: x["timestamp"], reverse=True)
    
    def delete_config(self, name: str) -> bool:
        """Delete configuration"""
        config_file = self.config_dir / f"{name}.json"
        
        if config_file.exists():
            config_file.unlink()
            return True
        
        return False
    
    def backup_configs(self, backup_path: str = None) -> str:
        """Backup all configurations"""
        if backup_path is None:
            backup_path = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        import zipfile
        
        with zipfile.ZipFile(backup_path, 'w') as zipf:
            for config_file in self.config_dir.glob("*.json"):
                zipf.write(config_file, config_file.name)
        
        return backup_path
    
    def restore_configs(self, backup_path: str) -> bool:
        """Restore configurations from backup"""
        try:
            import zipfile
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(self.config_dir)
            
            return True
        except Exception as e:
            print(f"Restore failed: {e}")
            return False

# Missing Dialog Classes - Fixed
class ConfigSaveDialog:
    """Dialog for saving configuration with name and description"""
    
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        self.dialog = None
    
    def show(self):
        """Show the dialog and return (name, description) or None"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Save Configuration")
        self.dialog.geometry("400x200")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (200 // 2)
        self.dialog.geometry(f"400x200+{x}+{y}")
        
        # Create form
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        ttk.Label(frame, text="Configuration Name:").pack(anchor="w")
        self.name_var = tk.StringVar(value=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        name_entry = ttk.Entry(frame, textvariable=self.name_var, width=40)
        name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Description field
        ttk.Label(frame, text="Description:").pack(anchor="w")
        self.desc_var = tk.StringVar()
        desc_entry = ttk.Entry(frame, textvariable=self.desc_var, width=40)
        desc_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Save", command=self._save).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)
        
        # Focus on name entry
        name_entry.focus()
        name_entry.select_range(0, tk.END)
        
        # Wait for dialog to close
        self.dialog.wait_window()
        return self.result
    
    def _save(self):
        """Handle save button"""
        name = self.name_var.get().strip()
        description = self.desc_var.get().strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter a configuration name.")
            return
        
        self.result = (name, description)
        self.dialog.destroy()
    
    def _cancel(self):
        """Handle cancel button"""
        self.result = None
        self.dialog.destroy()

class UserGuideDialog:
    """User guide dialog"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def show(self):
        """Show user guide"""
        guide_text = """
XAUUSD Multi-Timeframe EA - User Guide

GETTING STARTED:
1. Ensure MT5 is running and logged in
2. Select XAUUSD symbol in MT5
3. Click START to begin trading

ENTRY SIGNALS:
• BUY: Fractal Down + RSI > RSI_Upper (default: 55)
• SELL: Fractal Up + RSI < RSI_Lower (default: 45)

RECOVERY SYSTEM:
• Activates when position loses > Recovery_Price points
• Uses Martingale multiplication for lot sizing
• Smart Recovery waits for same signal before adding

PARAMETERS:
• Lot Size: Initial position size (0.01-10.0)
• RSI Upper/Lower: Signal thresholds (20-80)
• TP Points: Take profit in points (50-1000)
• Recovery Price: Loss threshold for recovery (50-500)
• Martingale: Lot multiplier for recovery (1.1-5.0)

RISK MANAGEMENT:
• Daily Loss Limit: Stop trading after daily loss
• Max Positions: Limit concurrent positions
• Max Drawdown: Emergency stop percentage

PRESETS:
• Scalping: Fast entries, small TP
• Intraday: Medium settings for day trading
• Swing: Larger TP, slower entries
• Conservative: Lower risk, higher thresholds

MONITOR:
• Watch Live Status for current metrics
• Check Risk panel for safety levels
• Review positions in Analysis tab

EMERGENCY FEATURES:
• Emergency Stop: Closes all positions immediately
• Auto-stop on risk limits exceeded
• Connection recovery with position sync

For support, check logs and error messages.
        """
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("User Guide")
        dialog.geometry("600x500")
        dialog.transient(self.parent)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"600x500+{x}+{y}")
        
        # Create text widget with scrollbar
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert guide text
        text_widget.insert(tk.END, guide_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        close_frame = ttk.Frame(dialog)
        close_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(close_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)

class AboutDialog:
    """About dialog"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def show(self):
        """Show about dialog"""
        about_text = """
XAUUSD Multi-Timeframe EA
Professional Trading System

Version: 1.0.0
Build: 2024.12.28

FEATURES:
✓ Multi-Timeframe Analysis
✓ Smart Recovery System  
✓ Real-time Risk Management
✓ Advanced UI with Live Charts
✓ Multi-Broker Compatibility
✓ Professional Logging System

STRATEGY:
• Fractal + RSI Entry Signals
• Dynamic Take Profit Calculation
• Anti-Hedge Protection
• Spread Management
• Position Correlation Analysis

TECHNOLOGY:
• Python 3.8+ with MT5 Integration
• Thread-safe Architecture
• SQLite Data Storage
• Real-time Performance Monitoring
• Advanced Error Handling

TRADING PAIRS:
• XAUUSD (Gold/USD) - Primary
• Auto-detection of symbol variations
• Support for different broker naming

RISK DISCLAIMER:
Trading involves significant risk of loss.
Past performance does not guarantee future results.
Only trade with money you can afford to lose.

© 2024 - Professional Trading Solutions
All rights reserved.
        """
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("About XAUUSD EA")
        dialog.geometry("500x400")
        dialog.transient(self.parent)
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"500x400+{x}+{y}")
        
        # Create main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Logo/Icon (placeholder)
        icon_frame = ttk.Frame(main_frame)
        icon_frame.pack(fill=tk.X, pady=(0, 10))
        
        icon_label = ttk.Label(icon_frame, text="🏆", font=("Arial", 24))
        icon_label.pack()
        
        # About text
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Segoe UI", 9), 
                             height=20, width=60)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert about text
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        ttk.Button(button_frame, text="OK", command=dialog.destroy).pack(side=tk.RIGHT)

# Enhanced UI Controller with advanced features
class AdvancedUIController:
    """Enhanced UI controller with advanced features"""
    
    def __init__(self, main_ui):
        self.main_ui = main_ui
        self.root = main_ui.root
        
        # Advanced components
        self.alert_manager = None
        self.performance_dashboard = None
        self.config_manager = ConfigurationManager()
        self.data_store = DataStore()
        
        # Initialize advanced features
        self.setup_advanced_features()
    
    def setup_advanced_features(self):
        """Setup advanced UI features"""
        try:
            # Add alert manager to main window
            alert_container = ttk.Frame(self.root)
            alert_container.pack(side=tk.TOP, fill=tk.X, before=self.main_ui.notebook)
            self.alert_manager = AlertManager(alert_container)
            
            # Add performance dashboard tab
            self.performance_dashboard = PerformanceDashboard(self.main_ui.advanced_frame)
            
            # Add advanced menu bar
            self.create_advanced_menu()
            
            # Add status bar
            self.create_status_bar()
        except Exception as e:
            print(f"Setup advanced features error: {e}")
    
    def create_advanced_menu(self):
        """Create advanced menu bar"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Save Configuration", command=self.save_configuration)
            file_menu.add_command(label="Load Configuration", command=self.load_configuration)
            file_menu.add_separator()
            file_menu.add_command(label="Export Logs", command=self.export_logs)
            file_menu.add_command(label="Export Data", command=self.export_data)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.main_ui.on_closing)
            
            # Tools menu
            tools_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Tools", menu=tools_menu)
            tools_menu.add_command(label="System Diagnostics", command=self.show_diagnostics)
            tools_menu.add_command(label="Performance Report", command=self.show_performance_report)
            tools_menu.add_command(label="Clear All Data", command=self.clear_all_data)
            
            # View menu
            view_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="View", menu=view_menu)
            view_menu.add_command(label="Show Alerts", command=self.show_alerts)
            view_menu.add_command(label="Performance Dashboard", command=self.show_performance_dashboard)
            view_menu.add_command(label="Full Screen", command=self.toggle_fullscreen)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="User Guide", command=self.show_user_guide)
            help_menu.add_command(label="About", command=self.show_about)
        except Exception as e:
            print(f"Create advanced menu error: {e}")
    
    def create_status_bar(self):
        """Create status bar"""
        try:
            self.status_bar = ttk.Frame(self.root)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Status sections
            self.status_text = tk.StringVar(value="Ready")
            ttk.Label(self.status_bar, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
            
            # Connection status
            self.connection_status = tk.StringVar(value="Disconnected")
            ttk.Label(self.status_bar, textvariable=self.connection_status).pack(side=tk.RIGHT, padx=5)
            
            # Memory usage
            self.memory_status = tk.StringVar(value="Memory: 0 MB")
            ttk.Label(self.status_bar, textvariable=self.memory_status).pack(side=tk.RIGHT, padx=5)
        except Exception as e:
            print(f"Create status bar error: {e}")
    
    def update_status(self, message: str):
        """Update status bar message"""
        try:
            self.status_text.set(message)
        except Exception as e:
            print(f"Update status error: {e}")
    
    def update_connection_status(self, connected: bool):
        """Update connection status"""
        try:
            status = "Connected" if connected else "Disconnected"
            self.connection_status.set(f"MT5: {status}")
        except Exception as e:
            print(f"Update connection status error: {e}")
    
    def update_memory_usage(self, memory_mb: float):
        """Update memory usage display"""
        try:
            self.memory_status.set(f"Memory: {memory_mb:.1f} MB")
        except Exception as e:
            print(f"Update memory usage error: {e}")
    
    # Alert methods
    def show_info_alert(self, title: str, message: str):
        """Show info alert"""
        try:
            if self.alert_manager:
                self.alert_manager.add_alert(AlertType.INFO, title, message)
        except Exception as e:
            print(f"Show info alert error: {e}")
    
    def show_success_alert(self, title: str, message: str):
        """Show success alert"""
        try:
            if self.alert_manager:
                self.alert_manager.add_alert(AlertType.SUCCESS, title, message)
        except Exception as e:
            print(f"Show success alert error: {e}")
    
    def show_warning_alert(self, title: str, message: str):
        """Show warning alert"""
        try:
            if self.alert_manager:
                self.alert_manager.add_alert(AlertType.WARNING, title, message, auto_dismiss=False)
        except Exception as e:
            print(f"Show warning alert error: {e}")
    
    def show_error_alert(self, title: str, message: str):
        """Show error alert"""
        try:
            if self.alert_manager:
                self.alert_manager.add_alert(AlertType.ERROR, title, message, auto_dismiss=False)
        except Exception as e:
            print(f"Show error alert error: {e}")
    
    def show_trade_alert(self, title: str, message: str):
        """Show trade alert"""
        try:
            if self.alert_manager:
                self.alert_manager.add_alert(AlertType.TRADE, title, message)
        except Exception as e:
            print(f"Show trade alert error: {e}")
    
    def show_signal_alert(self, title: str, message: str):
        """Show signal alert"""
        try:
            if self.alert_manager:
                self.alert_manager.add_alert(AlertType.SIGNAL, title, message)
        except Exception as e:
            print(f"Show signal alert error: {e}")
    
    # Configuration management
    def save_configuration(self):
        """Save current configuration"""
        try:
            # Get current config from main UI
            config = self.main_ui._collect_parameters() if hasattr(self.main_ui, '_collect_parameters') else {}
            
            # Ask for name and description
            dialog = ConfigSaveDialog(self.root)
            result = dialog.show()
            
            if result:
                name, description = result
                file_path = self.config_manager.save_config(config, name, description)
                self.show_success_alert("Configuration Saved", f"Saved to {file_path}")
            
        except Exception as e:
            self.show_error_alert("Save Error", f"Failed to save configuration: {e}")
    
    def load_configuration(self):
        """Load configuration"""
        try:
            configs = self.config_manager.list_configs()
            
            if not configs:
                self.show_info_alert("No Configurations", "No saved configurations found")
                return
            
            # Show config selection dialog
            messagebox.showinfo("Load Configuration", f"Found {len(configs)} saved configurations")
            
        except Exception as e:
            self.show_error_alert("Load Error", f"Failed to load configuration: {e}")
    
    def export_logs(self):
        """Export logs to file"""
        try:
            messagebox.showinfo("Export Logs", "Log export feature would be implemented here")
        except Exception as e:
            self.show_error_alert("Export Error", f"Failed to export logs: {e}")
    
    def export_data(self):
        """Export data to file"""
        try:
            messagebox.showinfo("Export Data", "Data export feature would be implemented here")
        except Exception as e:
            self.show_error_alert("Export Error", f"Failed to export data: {e}")
    
    def show_diagnostics(self):
        """Show system diagnostics"""
        try:
            messagebox.showinfo("System Diagnostics", "System diagnostics would be shown here")
        except Exception as e:
            self.show_error_alert("Diagnostics Error", f"Failed to show diagnostics: {e}")
    
    def show_performance_report(self):
        """Show performance report"""
        try:
            messagebox.showinfo("Performance Report", "Performance report would be shown here")
        except Exception as e:
            self.show_error_alert("Report Error", f"Failed to show performance report: {e}")
    
    def clear_all_data(self):
        """Clear all data"""
        try:
            if messagebox.askyesno("Confirm", "Clear all stored data?"):
                messagebox.showinfo("Clear Data", "All data would be cleared here")
        except Exception as e:
            self.show_error_alert("Clear Error", f"Failed to clear data: {e}")
    
    def show_alerts(self):
        """Show alerts panel"""
        try:
            messagebox.showinfo("Alerts", "Alerts panel is already visible")
        except Exception as e:
            print(f"Show alerts error: {e}")
    
    def show_performance_dashboard(self):
        """Show performance dashboard"""
        try:
            messagebox.showinfo("Performance Dashboard", "Switch to Advanced tab to view dashboard")
        except Exception as e:
            print(f"Show performance dashboard error: {e}")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        try:
            current_state = self.root.attributes('-fullscreen')
            self.root.attributes('-fullscreen', not current_state)
        except Exception as e:
            print(f"Toggle fullscreen error: {e}")
    
    def show_user_guide(self):
        """Show user guide"""
        try:
            dialog = UserGuideDialog(self.root)
            dialog.show()
        except Exception as e:
            self.show_error_alert("Guide Error", f"Failed to show user guide: {e}")
    
    def show_about(self):
        """Show about dialog"""
        try:
            dialog = AboutDialog(self.root)
            dialog.show()
        except Exception as e:
            self.show_error_alert("About Error", f"Failed to show about dialog: {e}")

# ============================================================================
# MAIN UI CONTROLLER (from ui_controller.py)
# ============================================================================


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
    from src.core.trading_engine import StrategyEngine, EngineState
    from src.core.trading_engine import TradingConfig
    from src.core.risk_manager import RiskLevel
    from src.core.position_manager import Position

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
        # self.logger.info("Engine initialization completed successfully")  # Commented for production
        
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
        self.logger.setLevel(logging.WARNING)
        
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
            from src.core.trading_engine import TradingConfig
            from src.core.trading_engine import StrategyEngine
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
            # self.logger.info("UI update thread started")  # Commented for production
    
    def stop_ui_updates(self):
        """Stop UI update thread"""
        self.running = False
        if self.update_thread:
            self.logger.info("UI update thread stopped")
    
    def ui_update_loop(self):
        """Enhanced UI update loop"""
        # self.logger.info("UI update loop started")  # Commented for production
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
                    from src.core.trading_engine import EngineState
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
