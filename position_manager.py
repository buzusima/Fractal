import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from trading_core import TradingConfig

@dataclass
class Position:
    """Enhanced position data structure"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    open_price: float
    open_time: datetime
    tp: float = 0.0
    sl: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    current_price: float = 0.0
    recovery_level: int = 0
    is_recovery: bool = False
    original_signal: str = ""
    comment: str = ""
    magic: int = 0
    
    # Calculated fields
    pips_profit: float = 0.0
    profit_points: float = 0.0
    
    def update_current_profit(self, current_price: float, point_value: float, contract_size: float):
        """Update current profit and related calculations"""
        self.current_price = current_price
        
        # Calculate profit in points
        if self.type == 0:  # BUY
            self.profit_points = (current_price - self.open_price) / point_value
        else:  # SELL
            self.profit_points = (self.open_price - current_price) / point_value
        
        # Calculate profit in currency
        self.profit = self.profit_points * point_value * self.volume * contract_size
        
        # Calculate pips (for display)
        self.pips_profit = self.profit_points / 10 if point_value == 0.00001 else self.profit_points

@dataclass
class RecoveryGroup:
    """Enhanced recovery group tracking"""
    group_id: str
    direction: int  # 0=BUY, 1=SELL
    positions: List[Position]
    total_volume: float = 0.0
    total_profit: float = 0.0
    avg_price: float = 0.0
    target_profit: float = 0.0
    current_level: int = 0
    max_level: int = 3
    last_signal_time: datetime = None
    creation_time: datetime = None
    break_even_price: float = 0.0
    required_tp_points: float = 0.0

class PositionManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions: Dict[int, Position] = {}
        self.recovery_groups: Dict[str, RecoveryGroup] = {}
        
        # Position tracking
        self.last_update = None
        self.update_count = 0
        
        # Symbol info cache
        self.symbol_info = None
        self.point_value = 0.0
        self.contract_size = 0.0
        
        # Exit speed multipliers by timeframe
        self.exit_speed_multipliers = {
            "M1": {"FAST": 0.5, "MEDIUM": 1.0, "SLOW": 2.0},
            "M5": {"FAST": 0.6, "MEDIUM": 1.2, "SLOW": 2.5},
            "M15": {"FAST": 0.8, "MEDIUM": 1.5, "SLOW": 3.0},
            "M30": {"FAST": 1.0, "MEDIUM": 2.0, "SLOW": 4.0},
            "H1": {"FAST": 1.2, "MEDIUM": 2.5, "SLOW": 5.0},
            "H4": {"FAST": 2.0, "MEDIUM": 4.0, "SLOW": 8.0},
            "D1": {"FAST": 3.0, "MEDIUM": 6.0, "SLOW": 12.0}
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize symbol info
        self._update_symbol_info()
    
    def _update_symbol_info(self):
        """Update symbol information for calculations"""
        try:
            self.symbol_info = mt5.symbol_info(self.config.symbol)
            if self.symbol_info:
                self.point_value = self.symbol_info.point
                self.contract_size = self.symbol_info.contract_size
                self.logger.debug(f"Symbol info updated: point={self.point_value}, contract={self.contract_size}")
            else:
                self.logger.warning("Could not get symbol info")
                # Fallback values for XAUUSD
                self.point_value = 0.01
                self.contract_size = 100.0
        except Exception as e:
            self.logger.error(f"Symbol info update error: {e}")
            # Fallback values
            self.point_value = 0.01
            self.contract_size = 100.0
    
    def update_positions(self) -> bool:
        """Enhanced position update with proper error handling"""
        try:
            # Get current positions from MT5
            mt5_positions = mt5.positions_get(symbol=self.config.symbol)
            if mt5_positions is None:
                mt5_positions = []
            
            # Convert to set for comparison
            current_tickets = {pos.ticket for pos in mt5_positions}
            tracked_tickets = set(self.positions.keys())
            
            # Remove closed positions
            closed_tickets = tracked_tickets - current_tickets
            for ticket in closed_tickets:
                self._on_position_closed(ticket)
            
            # Update existing positions and add new ones
            for mt5_pos in mt5_positions:
                if mt5_pos.ticket in self.positions:
                    self._update_existing_position(mt5_pos)
                else:
                    self._add_new_position(mt5_pos)
            
            # Update recovery groups
            self._update_recovery_groups()
            
            # Update tracking info
            self.last_update = datetime.now()
            self.update_count += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Position update error: {e}")
            return False
    
    def _add_new_position(self, mt5_pos):
        """Add new position with enhanced data"""
        try:
            # Create position object
            position = Position(
                ticket=mt5_pos.ticket,
                symbol=mt5_pos.symbol,
                type=mt5_pos.type,
                volume=mt5_pos.volume,
                open_price=mt5_pos.price_open,
                open_time=datetime.fromtimestamp(mt5_pos.time),
                tp=mt5_pos.tp,
                sl=mt5_pos.sl,
                profit=mt5_pos.profit,
                swap=mt5_pos.swap,
                commission=getattr(mt5_pos, 'commission', 0.0),
                comment=mt5_pos.comment,
                magic=mt5_pos.magic
            )
            
            # Update current profit calculations
            current_price = self._get_current_price(position.type)
            if current_price > 0:
                position.update_current_profit(current_price, self.point_value, self.contract_size)
            
            # Check if this is a recovery position
            self._identify_recovery_position(position)
            
            self.positions[mt5_pos.ticket] = position
            self.logger.info(f"Added position: {mt5_pos.ticket} {['BUY', 'SELL'][mt5_pos.type]} {mt5_pos.volume}")
            
        except Exception as e:
            self.logger.error(f"Add position error: {e}")
    
    def _update_existing_position(self, mt5_pos):
        """Update existing position with current data"""
        try:
            if mt5_pos.ticket not in self.positions:
                return
            
            position = self.positions[mt5_pos.ticket]
            
            # Update mutable fields
            position.profit = mt5_pos.profit
            position.swap = mt5_pos.swap
            position.tp = mt5_pos.tp
            position.sl = mt5_pos.sl
            
            # Update current profit calculations
            current_price = self._get_current_price(position.type)
            if current_price > 0:
                position.update_current_profit(current_price, self.point_value, self.contract_size)
            
        except Exception as e:
            self.logger.error(f"Update position error: {e}")
    
    def _get_current_price(self, position_type: int) -> float:
        """Get current price for position type"""
        try:
            tick = mt5.symbol_info_tick(self.config.symbol)
            if tick:
                return tick.bid if position_type == 0 else tick.ask
            return 0.0
        except:
            return 0.0
    
    def _identify_recovery_position(self, position: Position):
        """Identify if position is part of recovery system"""
        try:
            # Check comment for recovery indicators
            comment_lower = position.comment.lower()
            recovery_indicators = ['recovery', 'l1', 'l2', 'l3', 'l4', 'martingale']
            
            if any(indicator in comment_lower for indicator in recovery_indicators):
                position.is_recovery = True
                
                # Try to extract recovery level from comment
                for i in range(1, 10):
                    if f'l{i}' in comment_lower or f'level {i}' in comment_lower:
                        position.recovery_level = i
                        break
                
                self.logger.debug(f"Identified recovery position: {position.ticket} level {position.recovery_level}")
        
        except Exception as e:
            self.logger.error(f"Recovery identification error: {e}")
    
    def _on_position_closed(self, ticket: int):
        """Handle position closure"""
        try:
            if ticket not in self.positions:
                return
            
            position = self.positions[ticket]
            self.logger.info(f"Position closed: {ticket}, Final Profit: ${position.profit:.2f}")
            
            # Update recovery group if needed
            self._remove_from_recovery_group(position)
            
            # Remove from tracking
            del self.positions[ticket]
            
        except Exception as e:
            self.logger.error(f"Position closure handling error: {e}")
    
    def calculate_take_profit(self, direction: int, recovery_positions: List[Position] = None) -> float:
        """Enhanced TP calculation with proper validation"""
        try:
            if not recovery_positions or len(recovery_positions) == 0:
                # First position - use standard TP
                return self._get_base_tp_points()
            
            # Calculate TP for recovery group
            total_volume = sum(pos.volume for pos in recovery_positions)
            
            if total_volume <= 0:
                self.logger.warning("Invalid total volume in recovery group")
                return self._get_base_tp_points()
            
            # Target profit in USD (based on first position)
            first_position = recovery_positions[0]
            target_profit_usd = self._calculate_target_profit_usd(first_position.volume)
            
            # Calculate required TP points to achieve target profit
            # Profit = TP_points * point_value * total_volume * contract_size
            if self.point_value > 0 and self.contract_size > 0:
                required_tp_points = target_profit_usd / (self.point_value * total_volume * self.contract_size)
            else:
                self.logger.warning("Invalid point value or contract size")
                return self._get_base_tp_points()
            
            # Apply spread buffer
            spread_buffer = self._calculate_spread_buffer()
            tp_points = required_tp_points + spread_buffer
            
            # Apply exit speed modifier
            speed_modifier = self._get_exit_speed_modifier()
            tp_points *= speed_modifier
            
            # Validate result
            min_tp = 10  # Minimum 10 points TP
            max_tp = 1000  # Maximum 1000 points TP
            tp_points = max(min_tp, min(tp_points, max_tp))
            
            self.logger.debug(f"Calculated TP: {tp_points:.1f} points for {len(recovery_positions)} positions")
            
            return tp_points
            
        except Exception as e:
            self.logger.error(f"TP calculation error: {e}")
            return self._get_base_tp_points()
    
    def _get_base_tp_points(self) -> float:
        """Get base TP points for first position"""
        try:
            speed_modifier = self._get_exit_speed_modifier()
            base_tp = self.config.tp_first * speed_modifier
            return max(10, min(base_tp, 1000))  # Validate range
        except:
            return 200  # Default fallback
    
    def _calculate_target_profit_usd(self, base_volume: float) -> float:
        """Calculate target profit in USD for recovery"""
        try:
            # Standard XAUUSD: $0.01 per point per 0.01 lot
            # Target profit = TP_points * volume * point_value_per_lot
            base_tp_points = self.config.tp_first
            point_value_per_lot = self.point_value * self.contract_size  # Should be ~$1 per point per lot
            target_profit = base_tp_points * base_volume * point_value_per_lot
            
            self.logger.debug(f"Target profit: ${target_profit:.2f} for {base_volume} lots")
            return target_profit
            
        except Exception as e:
            self.logger.error(f"Target profit calculation error: {e}")
            return 20.0  # Default $20 target
    
    def _get_exit_speed_modifier(self) -> float:
        """Get exit speed modifier based on timeframe and setting"""
        try:
            tf = self.config.primary_tf
            speed_names = ["FAST", "MEDIUM", "SLOW"]
            
            if 0 <= self.config.exit_speed < len(speed_names):
                speed_name = speed_names[self.config.exit_speed]
            else:
                speed_name = "MEDIUM"
            
            multipliers = self.exit_speed_multipliers.get(tf, self.exit_speed_multipliers["M15"])
            modifier = multipliers.get(speed_name, 1.0)
            
            return max(0.1, min(modifier, 10.0))  # Validate range
            
        except Exception as e:
            self.logger.error(f"Exit speed modifier error: {e}")
            return 1.0
    
    def _calculate_spread_buffer(self) -> int:
        """Calculate spread buffer for TP"""
        try:
            symbol_info = mt5.symbol_info(self.config.symbol)
            if symbol_info is None:
                return 5
            
            current_spread = symbol_info.spread
            
            if self.config.spread_mode == 0:  # AUTO
                buffer = int(current_spread * 1.5) + 2
            elif self.config.spread_mode == 1:  # FIXED
                buffer = self.config.spread_buffer
            elif self.config.spread_mode == 2:  # SMART
                buffer = int(current_spread * 1.2) + 1
            else:  # NONE
                buffer = 0
            
            return max(0, min(buffer, 50))  # Cap at 50 points
            
        except Exception as e:
            self.logger.error(f"Spread buffer calculation error: {e}")
            return 5
    
    def check_recovery_needed(self, position: Position) -> bool:
        """Enhanced recovery check with proper validation"""
        try:
            # Don't check recovery for already recovered positions
            if position.is_recovery:
                return False
            
            # Check if profit is negative
            if position.profit >= 0:
                return False
            
            # Calculate loss in points using current market price
            current_price = self._get_current_price(position.type)
            if current_price <= 0:
                self.logger.warning("Could not get current price for recovery check")
                return False
            
            # Calculate price difference
            if position.type == 0:  # BUY position
                price_diff = position.open_price - current_price  # Loss when price goes down
            else:  # SELL position
                price_diff = current_price - position.open_price  # Loss when price goes up
            
            # Convert to points
            if self.point_value > 0:
                loss_points = price_diff / self.point_value
            else:
                self.logger.warning("Invalid point value for recovery calculation")
                return False
            
            # Check if loss exceeds recovery threshold
            recovery_needed = loss_points >= self.config.recovery_price
            
            if recovery_needed:
                self.logger.info(f"Recovery needed for position {position.ticket}: {loss_points:.1f} points loss")
            
            return recovery_needed
            
        except Exception as e:
            self.logger.error(f"Recovery check error: {e}")
            return False
    
    def get_recovery_lot_size(self, original_volume: float, recovery_level: int) -> float:
        """Calculate lot size for recovery position with validation"""
        try:
            if recovery_level <= 0:
                return original_volume
            
            multiplier = self.config.martingale ** recovery_level
            recovery_volume = original_volume * multiplier
            
            # Validate against symbol limits
            if self.symbol_info:
                min_vol = self.symbol_info.volume_min
                max_vol = self.symbol_info.volume_max
                step = self.symbol_info.volume_step
                
                # Round to valid step
                recovery_volume = round(recovery_volume / step) * step
                
                # Apply limits
                recovery_volume = max(min_vol, min(recovery_volume, max_vol))
            else:
                # Basic validation without symbol info
                recovery_volume = max(0.01, min(recovery_volume, 10.0))
                recovery_volume = round(recovery_volume, 2)
            
            self.logger.debug(f"Recovery volume: {recovery_volume} (level {recovery_level}, multiplier {multiplier:.2f})")
            
            return recovery_volume
            
        except Exception as e:
            self.logger.error(f"Recovery lot size calculation error: {e}")
            return original_volume
    
    def can_add_recovery(self, group_id: str) -> bool:
        """Check if can add recovery position to group"""
        try:
            if group_id not in self.recovery_groups:
                return True
            
            group = self.recovery_groups[group_id]
            can_add = group.current_level < self.config.max_recovery
            
            if not can_add:
                self.logger.info(f"Max recovery level reached for group {group_id}")
            
            return can_add
            
        except Exception as e:
            self.logger.error(f"Recovery check error: {e}")
            return False
    
    def create_recovery_group(self, original_position: Position) -> str:
        """Create recovery group for position"""
        try:
            group_id = f"{original_position.type}_{original_position.open_time.strftime('%Y%m%d_%H%M%S')}_{original_position.ticket}"
            
            group = RecoveryGroup(
                group_id=group_id,
                direction=original_position.type,
                positions=[original_position],
                total_volume=original_position.volume,
                target_profit=self._calculate_target_profit_usd(original_position.volume),
                max_level=self.config.max_recovery,
                creation_time=datetime.now()
            )
            
            self.recovery_groups[group_id] = group
            original_position.recovery_level = 0
            original_position.is_recovery = False  # Original position is not recovery
            
            self.logger.info(f"Created recovery group: {group_id}")
            return group_id
            
        except Exception as e:
            self.logger.error(f"Recovery group creation error: {e}")
            return ""
    
    def add_to_recovery_group(self, group_id: str, position: Position):
        """Add position to recovery group"""
        try:
            if group_id not in self.recovery_groups:
                self.logger.error(f"Recovery group {group_id} not found")
                return
            
            group = self.recovery_groups[group_id]
            group.positions.append(position)
            group.total_volume += position.volume
            group.current_level += 1
            
            position.recovery_level = group.current_level
            position.is_recovery = True
            
            self.logger.info(f"Added to recovery group {group_id}: Level {group.current_level}")
            
        except Exception as e:
            self.logger.error(f"Add to recovery group error: {e}")
    
    def _update_recovery_groups(self):
        """Update recovery group statistics"""
        try:
            for group_id, group in self.recovery_groups.items():
                # Update total profit
                group.total_profit = sum(pos.profit for pos in group.positions)
                
                # Calculate weighted average price
                if group.total_volume > 0:
                    weighted_sum = sum(pos.open_price * pos.volume for pos in group.positions)
                    group.avg_price = weighted_sum / group.total_volume
                
                # Calculate break-even price (considering spread)
                if group.total_volume > 0:
                    current_spread = self._calculate_spread_buffer()
                    spread_adjustment = current_spread * self.point_value
                    
                    if group.direction == 0:  # BUY group
                        group.break_even_price = group.avg_price + spread_adjustment
                    else:  # SELL group
                        group.break_even_price = group.avg_price - spread_adjustment
                
                # Calculate required TP points for target profit
                if group.total_volume > 0 and self.point_value > 0 and self.contract_size > 0:
                    group.required_tp_points = group.target_profit / (self.point_value * group.total_volume * self.contract_size)
                
        except Exception as e:
            self.logger.error(f"Recovery groups update error: {e}")
    
    def _remove_from_recovery_group(self, position: Position):
        """Remove position from recovery group when closed"""
        try:
            for group_id, group in list(self.recovery_groups.items()):
                if position in group.positions:
                    group.positions.remove(position)
                    group.total_volume -= position.volume
                    
                    # If group is empty, remove it
                    if not group.positions:
                        del self.recovery_groups[group_id]
                        self.logger.info(f"Recovery group completed: {group_id}")
                    else:
                        # Recalculate group statistics
                        self._update_recovery_groups()
                    break
                    
        except Exception as e:
            self.logger.error(f"Remove from recovery group error: {e}")
    
    def get_position_summary(self) -> Dict:
        """Get enhanced summary of all positions"""
        try:
            total_profit = sum(pos.profit for pos in self.positions.values())
            buy_positions = [pos for pos in self.positions.values() if pos.type == 0]
            sell_positions = [pos for pos in self.positions.values() if pos.type == 1]
            recovery_positions = [pos for pos in self.positions.values() if pos.is_recovery]
            
            summary = {
                "total_positions": len(self.positions),
                "buy_positions": len(buy_positions),
                "sell_positions": len(sell_positions),
                "recovery_positions": len(recovery_positions),
                "total_profit": total_profit,
                "total_volume": sum(pos.volume for pos in self.positions.values()),
                "recovery_groups": len(self.recovery_groups),
                "buy_profit": sum(pos.profit for pos in buy_positions),
                "sell_profit": sum(pos.profit for pos in sell_positions),
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "update_count": self.update_count
            }
            
            # Add position details
            summary["positions"] = []
            for pos in self.positions.values():
                summary["positions"].append({
                    "ticket": pos.ticket,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "open_price": pos.open_price,
                    "current_price": pos.current_price,
                    "profit": pos.profit,
                    "profit_points": pos.profit_points,
                    "is_recovery": pos.is_recovery,
                    "recovery_level": pos.recovery_level,
                    "open_time": pos.open_time.isoformat()
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Position summary error: {e}")
            return {"error": str(e)}
    
    def get_recovery_status(self) -> Dict:
        """Get detailed recovery groups status"""
        try:
            status = {}
            for group_id, group in self.recovery_groups.items():
                status[group_id] = {
                    "direction": "BUY" if group.direction == 0 else "SELL",
                    "level": group.current_level,
                    "max_level": group.max_level,
                    "total_volume": group.total_volume,
                    "total_profit": group.total_profit,
                    "target_profit": group.target_profit,
                    "avg_price": group.avg_price,
                    "break_even_price": group.break_even_price,
                    "required_tp_points": group.required_tp_points,
                    "positions_count": len(group.positions),
                    "creation_time": group.creation_time.isoformat() if group.creation_time else None,
                    "positions": [
                        {
                            "ticket": pos.ticket,
                            "volume": pos.volume,
                            "open_price": pos.open_price,
                            "profit": pos.profit,
                            "recovery_level": pos.recovery_level
                        }
                        for pos in group.positions
                    ]
                }
            return status
            
        except Exception as e:
            self.logger.error(f"Recovery status error: {e}")
            return {"error": str(e)}
    
    def emergency_close_all(self) -> List[int]:
        """Emergency close all positions with proper error handling"""
        closed_tickets = []
        
        try:
            self.logger.warning("EMERGENCY CLOSE ALL POSITIONS INITIATED")
            
            for ticket, position in list(self.positions.items()):
                try:
                    # Determine close order type
                    close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
                    
                    # Get current price
                    current_price = self._get_current_price(1 - position.type)  # Opposite of position type
                    if current_price <= 0:
                        self.logger.error(f"Cannot get price for closing position {ticket}")
                        continue
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": ticket,
                        "symbol": self.config.symbol,
                        "volume": position.volume,
                        "type": close_type,
                        "price": current_price,
                        "deviation": 50,  # Allow higher deviation for emergency close
                        "magic": 234000,
                        "comment": "Emergency close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed_tickets.append(ticket)
                        self.logger.info(f"Emergency closed position: {ticket}")
                    else:
                        error_msg = f"retcode={result.retcode}" if result else "no result"
                        self.logger.error(f"Failed to close position {ticket}: {error_msg}")
                        
                except Exception as e:
                    self.logger.error(f"Error closing position {ticket}: {e}")
            
            self.logger.warning(f"Emergency close completed: {len(closed_tickets)} positions closed")
            return closed_tickets
            
        except Exception as e:
            self.logger.error(f"Emergency close all error: {e}")
            return closed_tickets