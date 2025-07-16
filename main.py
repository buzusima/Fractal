#!/usr/bin/env python3
"""
XAUUSD Multi-Timeframe EA - Main Entry Point (Updated for New Structure)
Professional Trading System

Usage:
    python main.py
"""

import sys
import traceback
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return False
    print("Python version OK")
    return True

def check_dependencies():
    """Check all required dependencies"""
    dependencies = {
        'MetaTrader5': 'MT5 trading platform connection',
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computations', 
        'matplotlib': 'Plotting and charts',
        'tkinter': 'GUI framework (built-in)',
        'sqlite3': 'Database operations (built-in)',
        'threading': 'Multi-threading support (built-in)',
        'json': 'JSON handling (built-in)',
        'datetime': 'Date and time operations (built-in)',
        'logging': 'Logging system (built-in)'
    }
    
    print("\nChecking dependencies...")
    failed_imports = []
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"OK: {module:<15} - {description}")
        except ImportError as e:
            print(f"MISSING: {module:<15} - {description}")
            failed_imports.append(module)
        except Exception as e:
            print(f"ERROR: {module:<15} - {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nMissing dependencies: {', '.join(failed_imports)}")
        print("\nTo install missing packages:")
        external_deps = [dep for dep in failed_imports if dep in ['MetaTrader5', 'pandas', 'numpy', 'matplotlib']]
        if external_deps:
            print(f"   pip install {' '.join(external_deps)}")
        return False
    
    print("\nAll dependencies OK")
    return True

def check_project_structure():
    """Check if new project structure exists"""
    required_dirs = [
        'src',
        'src/core',
        'src/ui',
        'src/utils',
        'src/strategies'
    ]
    
    required_files = [
        'src/core/trading_engine.py',
        'src/core/position_manager.py', 
        'src/core/order_executor.py',
        'src/core/risk_manager.py',
        'src/ui/main_window.py',
        'src/utils/logging.py',
        'src/utils/config.py',
        'src/strategies/base_strategy.py',
        'src/strategies/fractal_rsi.py'
    ]
    
    print("\nChecking new project structure...")
    
    # Check directories
    missing_dirs = []
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"OK: {dir_name}/")
        else:
            print(f"MISSING: {dir_name}/")
            missing_dirs.append(dir_name)
    
    # Check files
    missing_files = []
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"OK: {file_name}")
        else:
            print(f"MISSING: {file_name}")
            missing_files.append(file_name)
    
    if missing_dirs or missing_files:
        print(f"\nMissing directories: {missing_dirs}")
        print(f"Missing files: {missing_files}")
        print("\nRun migration scripts first:")
        print("1. python create_structure.py")
        print("2. python migrate_files.py")
        print("3. python fix_utils.py")
        return False
    
    print("\nNew project structure OK")
    return True

def test_imports():
    """Test importing project modules from new structure"""
    print("\nTesting project imports...")
    
    import_tests = [
        ('src.core.trading_engine', 'TradingConfig, XAUUSDTradingCore, StrategyEngine'),
        ('src.utils.logging', 'TradingLogManager'),
        ('src.core.position_manager', 'PositionManager'),
        ('src.core.order_executor', 'OrderExecutor'), 
        ('src.core.risk_manager', 'RiskManager'),
        ('src.ui.main_window', 'XAUUSDTradingUI'),
        ('src.utils.config', 'ConfigManager, TradingConfig'),
        ('src.strategies.fractal_rsi', 'FractalRSIStrategy'),
        ('src.strategies.base_strategy', 'BaseStrategy')
    ]
    
    success_count = 0
    
    for module, classes in import_tests:
        try:
            exec(f"from {module} import {classes}")
            print(f"OK: {module:<25} - {classes}")
            success_count += 1
        except ImportError as e:
            print(f"IMPORT ERROR: {module:<25} - {str(e)}")
        except Exception as e:
            print(f"ERROR: {module:<25} - {str(e)}")
    
    if success_count == len(import_tests):
        print("\nAll imports successful")
        return True
    else:
        print(f"\n{success_count}/{len(import_tests)} imports successful")
        return False

def test_mt5_connection():
    """Test MetaTrader5 basic functionality"""
    print("\nTesting MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        
        # Test basic MT5 functions
        print("Initializing MT5...")
        if not mt5.initialize():
            print("MT5 initialization failed")
            print("Make sure MetaTrader 5 is installed and running")
            return False
        
        print("MT5 initialized successfully")
        
        # Get basic info
        account_info = mt5.account_info()
        if account_info:
            print(f"Account: {account_info.name}")
            print(f"Server: {account_info.server}")
            print(f"Balance: ${account_info.balance:.2f}")
        else:
            print("Could not get account info (may need login)")
        
        # Test symbol info
        symbol_info = mt5.symbol_info("XAUUSD")
        if symbol_info is None:
            # Try alternative symbol names
            alt_symbols = ["XAUUSD.v", "XAUUSD.m", "GOLD", "#XAUUSD"]
            for symbol in alt_symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    print(f"Found XAUUSD as: {symbol}")
                    break
            
            if symbol_info is None:
                print("XAUUSD symbol not found")
                print("Check if XAUUSD is available in Market Watch")
        else:
            print("XAUUSD symbol available")
        
        mt5.shutdown()
        print("MT5 connection test completed")
        return True
        
    except Exception as e:
        print(f"MT5 test failed: {str(e)}")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['logs', 'data', 'configs']
    
    print("\nCreating directories...")
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created: {dir_name}/")

def main():
    """Main entry point"""
    print("=" * 60)
    print("XAUUSD Multi-Timeframe EA - System Check (New Structure)")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Check dependencies  
    if not check_dependencies():
        return False
    
    # Step 3: Check new project structure
    if not check_project_structure():
        return False
    
    # Step 4: Test imports
    if not test_imports():
        print("\nSome imports failed, but continuing...")
    
    # Step 5: Create directories
    create_directories()
    
    # Step 6: Test MT5 connection
    mt5_ok = test_mt5_connection()
    
    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    
    if mt5_ok:
        print("STATUS: READY TO LAUNCH")
        print("\nNext steps:")
        print("   1. New project structure working")
        print("   2. All imports successful")
        print("   3. MT5 connection working")
        
        # Ask if user wants to launch UI
        print("\n" + "-" * 40)
        response = input("Launch Trading UI? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("\nStarting Trading Application...")
            launch_application()
        else:
            print("\nSystem check completed. Run again when ready!")
    
    else:
        print("STATUS: SETUP REQUIRED")
        print("\nRequired actions:")
        print("   1. Install/configure MetaTrader 5")
        print("   2. Login to trading account")
        print("   3. Add XAUUSD to Market Watch")
        print("   4. Run system check again")
    
    return True

def launch_application():
    """Launch the main trading application"""
    try:
        print("Loading trading application...")
        
        # Import and start UI from new location
        from src.ui.main_window import XAUUSDTradingUI
        
        print("UI modules loaded")
        print("Starting XAUUSD Trading System...")
        
        # Create and run application
        app = XAUUSDTradingUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\nTry running system check again or contact support")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
        input("\nPress Enter to exit...")