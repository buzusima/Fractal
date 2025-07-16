# XAUUSD Multi-Timeframe EA

Professional automated trading system for XAUUSD (Gold/USD) using Fractal + RSI strategy with Smart Recovery system.

## Features

- **Multi-timeframe Analysis**: Supports M1, M5, M15, M30, H1, H4, D1
- **Smart Recovery System**: Intelligent Martingale-based recovery
- **Real-time UI**: Professional trading interface
- **Risk Management**: Comprehensive risk controls
- **Thread-safe Architecture**: Robust multi-threaded design

## Installation

1. Install Python 3.8+
2. Install MetaTrader 5
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python main.py
```

## Configuration

Edit `config.json` to customize trading parameters.

## Project Structure

```
xauusd_ea/
├── main.py              # Entry point
├── config.json          # Configuration
├── src/
│   ├── core/            # Core trading logic
│   ├── ui/              # User interface
│   ├── utils/           # Utilities
│   └── strategies/      # Trading strategies
├── tests/               # Unit tests
├── logs/                # Log files
└── docs/                # Documentation
```

## Warning

This system uses Martingale strategy which carries high risk. Only use with money you can afford to lose.

## License

Professional Trading System - All rights reserved.
