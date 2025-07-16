import MetaTrader5 as mt5
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from trading_core import TradingConfig
from position_manager import PositionManager, Position

class OrderType(Enum):
    MARKET_BUY = 0
    MARKET_SELL = 1
    PENDING_BUY = 2
    PENDING_SELL = 3
    CLOSE_BUY = 4
    CLOSE_SELL = 5

class ExecutionMode(Enum):
    MARKET = "market"
    INSTANT = "instant"
    EXCHANGE = "exchange"
    REQUEST = "request"

@dataclass
class OrderRequest:
    """Order request structure"""
    action: str
    symbol: str
    volume: float
    type: int
    price: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    deviation: int = 20
    comment: str = ""
    magic: int = 234000
    is_recovery: bool = False
    recovery_level: int = 0
    original_signal: str = ""

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    ticket: int = 0
    retcode: int = 0
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0
    error_msg: str = ""

class OrderExecutor:
    def __init__(self, config: TradingConfig, position_manager: PositionManager):
        self.config = config
        self.position_manager = position_manager
        self.execution_mode = ExecutionMode.MARKET
        self.max_slippage = 20
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Order tracking
        self.pending_orders = {}
        self.execution_history = []
        
        # Performance tracking
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.average_execution_time = 0.0
        
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect broker capabilities
        self._detect_execution_mode()
    
    def _detect_execution_mode(self):
        """Auto-detect broker execution capabilities with safe constant handling"""
        try:
            symbol_info = mt5.symbol_info(self.config.symbol)
            if symbol_info is None:
                self.execution_mode = ExecutionMode.MARKET
                return
            
            # Get filling mode safely
            filling_mode = getattr(symbol_info, 'filling_mode', 1)
            
            # Check filling modes using safe constant access
            # Use numeric values as fallback if constants don't exist
            ioc_mode = getattr(mt5, 'SYMBOL_FILLING_IOC', 1)  # 1 is typical IOC value
            fok_mode = getattr(mt5, 'SYMBOL_FILLING_FOK', 2)  # 2 is typical FOK value
            return_mode = getattr(mt5, 'SYMBOL_FILLING_RETURN', 4)  # 4 is typical RETURN value
            
            if filling_mode & ioc_mode:
                self.execution_mode = ExecutionMode.MARKET
            elif filling_mode & fok_mode:
                self.execution_mode = ExecutionMode.INSTANT
            elif filling_mode & return_mode:
                self.execution_mode = ExecutionMode.REQUEST
            else:
                # Default to market execution
                self.execution_mode = ExecutionMode.MARKET
            
            self.logger.info(f"Detected execution mode: {self.execution_mode.value}")
            self.logger.info(f"Filling mode value: {filling_mode}")
            
        except Exception as e:
            self.logger.error(f"Execution mode detection error: {e}")
            self.execution_mode = ExecutionMode.MARKET

    def execute_market_order(self, order_type: OrderType, volume: float, 
                           tp_points: float = 0, sl_points: float = 0,
                           comment: str = "", is_recovery: bool = False) -> OrderResult:
        """Execute market order with retry logic"""
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Get current prices
                symbol_info = mt5.symbol_info_tick(self.config.symbol)
                if symbol_info is None:
                    return OrderResult(False, error_msg="Failed to get symbol info")
                
                # Determine order direction and price
                if order_type == OrderType.MARKET_BUY:
                    price = symbol_info.ask
                    order_type_mt5 = mt5.ORDER_TYPE_BUY
                elif order_type == OrderType.MARKET_SELL:
                    price = symbol_info.bid
                    order_type_mt5 = mt5.ORDER_TYPE_SELL
                else:
                    return OrderResult(False, error_msg="Invalid order type for market execution")
                
                # Calculate TP and SL prices
                tp_price = self._calculate_tp_price(order_type, price, tp_points)
                sl_price = self._calculate_sl_price(order_type, price, sl_points)
                
                # Validate volume
                volume = self._validate_volume(volume)
                if volume <= 0:
                    return OrderResult(False, error_msg="Invalid volume")
                
                # Create order request
                request = self._create_order_request(
                    action=mt5.TRADE_ACTION_DEAL,
                    type=order_type_mt5,
                    volume=volume,
                    price=price,
                    tp=tp_price,
                    sl=sl_price,
                    comment=comment
                )
                
                # Execute order
                result = mt5.order_send(request)
                
                # Process result
                execution_time = time.time() - start_time
                order_result = self._process_order_result(result, execution_time)
                
                if order_result.success:
                    self._log_successful_order(order_result, order_type, volume, comment, is_recovery)
                    return order_result
                else:
                    self.logger.warning(f"Order attempt {attempt + 1} failed: {order_result.error_msg}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        self._log_failed_order(order_result, order_type, volume, comment)
                        return order_result
                        
            except Exception as e:
                error_msg = f"Order execution error: {str(e)}"
                self.logger.error(error_msg)
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return OrderResult(False, error_msg=error_msg)
        
        return OrderResult(False, error_msg="Max retries exceeded")
    
    def close_position(self, position: Position) -> OrderResult:
        """Close specific position"""
        try:
            # Determine close order type
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            
            # Get current price
            symbol_info = mt5.symbol_info_tick(self.config.symbol)
            if symbol_info is None:
                return OrderResult(False, error_msg="Failed to get current price")
            
            price = symbol_info.bid if position.type == 0 else symbol_info.ask
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": self.config.symbol,
                "volume": position.volume,
                "type": close_type,
                "price": price,
                "deviation": self.max_slippage,
                "magic": 234000,
                "comment": f"Close position {position.ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_mode(),
            }
            
            result = mt5.order_send(request)
            execution_time = time.time()
            
            order_result = self._process_order_result(result, 0)
            
            if order_result.success:
                self.logger.info(f"Position closed: {position.ticket}, Profit: {position.profit}")
            else:
                self.logger.error(f"Failed to close position {position.ticket}: {order_result.error_msg}")
            
            return order_result
            
        except Exception as e:
            error_msg = f"Close position error: {str(e)}"
            self.logger.error(error_msg)
            return OrderResult(False, error_msg=error_msg)
    
    def close_recovery_group(self, group_id: str) -> List[OrderResult]:
        """Close all positions in recovery group"""
        results = []
        
        if group_id not in self.position_manager.recovery_groups:
            return results
        
        group = self.position_manager.recovery_groups[group_id]
        
        for position in group.positions:
            if position.ticket in self.position_manager.positions:
                result = self.close_position(position)
                results.append(result)
        
        return results
    
    def modify_position_tp(self, ticket: int, new_tp: float) -> OrderResult:
        """Modify position take profit"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return OrderResult(False, error_msg="Position not found")
            
            pos = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": pos.sl,
                "tp": new_tp,
                "magic": 234000,
                "comment": f"Modify TP: {new_tp}",
            }
            
            result = mt5.order_send(request)
            order_result = self._process_order_result(result, 0)
            
            if order_result.success:
                self.logger.info(f"TP modified for position {ticket}: {new_tp}")
            else:
                self.logger.error(f"Failed to modify TP for {ticket}: {order_result.error_msg}")
            
            return order_result
            
        except Exception as e:
            error_msg = f"Modify TP error: {str(e)}"
            self.logger.error(error_msg)
            return OrderResult(False, error_msg=error_msg)
    
    def execute_recovery_order(self, original_position: Position, group_id: str) -> OrderResult:
        """Execute recovery order based on original signal"""
        # Check if recovery is allowed
        if not self.position_manager.can_add_recovery(group_id):
            return OrderResult(False, error_msg="Max recovery level reached")
        
        # Get recovery level and lot size
        recovery_level = self.position_manager.recovery_groups[group_id].current_level + 1
        recovery_volume = self.position_manager.get_recovery_lot_size(
            original_position.volume, recovery_level
        )
        
        # Wait for same signal if smart recovery is enabled
        if self.config.smart_recovery:
            if not self._wait_for_recovery_signal(original_position.type):
                return OrderResult(False, error_msg="Recovery signal not confirmed")
        
        # Calculate TP for recovery group
        recovery_positions = self.position_manager.recovery_groups[group_id].positions
        recovery_tp = self.position_manager.calculate_take_profit(
            original_position.type, recovery_positions
        )
        
        # Execute recovery order
        order_type = OrderType.MARKET_BUY if original_position.type == 0 else OrderType.MARKET_SELL
        comment = f"Recovery L{recovery_level} for {original_position.ticket}"
        
        result = self.execute_market_order(
            order_type=order_type,
            volume=recovery_volume,
            tp_points=recovery_tp,
            comment=comment,
            is_recovery=True
        )
        
        if result.success:
            # Update recovery group
            new_position = Position(
                ticket=result.ticket,
                symbol=self.config.symbol,
                type=original_position.type,
                volume=recovery_volume,
                open_price=result.price,
                open_time=datetime.now(),
                tp=self._calculate_tp_price(order_type, result.price, recovery_tp),
                recovery_level=recovery_level,
                is_recovery=True,
                original_signal=original_position.original_signal
            )
            
            self.position_manager.add_to_recovery_group(group_id, new_position)
            
            # Update TP for all positions in group
            self._update_group_take_profits(group_id)
        
        return result
    
    def _wait_for_recovery_signal(self, original_type: int) -> bool:
        """Wait for same entry signal for smart recovery"""
        # This would integrate with trading_core signal analysis
        # For now, return True (implement signal checking logic)
        return True
    
    def _update_group_take_profits(self, group_id: str):
        """Update take profits for all positions in recovery group"""
        if group_id not in self.position_manager.recovery_groups:
            return
        
        group = self.position_manager.recovery_groups[group_id]
        new_tp = self.position_manager.calculate_take_profit(group.direction, group.positions)
        
        for position in group.positions:
            if position.ticket in self.position_manager.positions:
                tp_price = self._calculate_tp_price(
                    OrderType.MARKET_BUY if position.type == 0 else OrderType.MARKET_SELL,
                    position.open_price, new_tp
                )
                self.modify_position_tp(position.ticket, tp_price)
    
    def _create_order_request(self, action: int, type: int, volume: float,
                             price: float = 0, tp: float = 0, sl: float = 0,
                             comment: str = "") -> Dict:
        """Create MT5 order request"""
        request = {
            "action": action,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.max_slippage,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }
        
        return request
    
    def _get_filling_mode(self) -> int:
        """Get appropriate filling mode for broker"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_IOC
        
        if symbol_info.filling_mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        elif symbol_info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        else:
            return mt5.ORDER_FILLING_RETURN
    
    def _validate_volume(self, volume: float) -> float:
        """Validate and normalize volume"""
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return 0
        
        # Check minimum volume
        if volume < symbol_info.volume_min:
            volume = symbol_info.volume_min
        
        # Check maximum volume
        if volume > symbol_info.volume_max:
            volume = symbol_info.volume_max
        
        # Round to volume step
        volume_step = symbol_info.volume_step
        volume = round(volume / volume_step) * volume_step
        
        return volume
    
    def _calculate_tp_price(self, order_type: OrderType, entry_price: float, tp_points: float) -> float:
        """Calculate TP price from points"""
        if tp_points <= 0:
            return 0
        
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return 0
        
        point = symbol_info.point
        
        if order_type in [OrderType.MARKET_BUY, OrderType.PENDING_BUY]:
            return entry_price + (tp_points * point)
        else:
            return entry_price - (tp_points * point)
    
    def _calculate_sl_price(self, order_type: OrderType, entry_price: float, sl_points: float) -> float:
        """Calculate SL price from points"""
        if sl_points <= 0:
            return 0
        
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            return 0
        
        point = symbol_info.point
        
        if order_type in [OrderType.MARKET_BUY, OrderType.PENDING_BUY]:
            return entry_price - (sl_points * point)
        else:
            return entry_price + (sl_points * point)
    
    def _process_order_result(self, mt5_result, execution_time: float) -> OrderResult:
        """Process MT5 order result"""
        if mt5_result is None:
            return OrderResult(False, error_msg="No result from MT5")
        
        success = mt5_result.retcode == mt5.TRADE_RETCODE_DONE
        
        result = OrderResult(
            success=success,
            ticket=mt5_result.order if hasattr(mt5_result, 'order') else 0,
            retcode=mt5_result.retcode,
            deal=mt5_result.deal if hasattr(mt5_result, 'deal') else 0,
            order=mt5_result.order if hasattr(mt5_result, 'order') else 0,
            volume=mt5_result.volume if hasattr(mt5_result, 'volume') else 0,
            price=mt5_result.price if hasattr(mt5_result, 'price') else 0,
            bid=mt5_result.bid if hasattr(mt5_result, 'bid') else 0,
            ask=mt5_result.ask if hasattr(mt5_result, 'ask') else 0,
            comment=mt5_result.comment if hasattr(mt5_result, 'comment') else "",
            request_id=mt5_result.request_id if hasattr(mt5_result, 'request_id') else 0
        )
        
        if not success:
            result.error_msg = f"MT5 Error {mt5_result.retcode}: {self._get_error_description(mt5_result.retcode)}"
        
        # Update performance metrics
        self.total_orders += 1
        if success:
            self.successful_orders += 1
            self._update_execution_time(execution_time)
        else:
            self.failed_orders += 1
        
        return result
    
    def _get_error_description(self, retcode: int) -> str:
        """Get error description for MT5 return code"""
        error_dict = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Request partially completed",
            mt5.TRADE_RETCODE_ERROR: "Request processing error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "No money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "Off quotes",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid expiration",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
            mt5.TRADE_RETCODE_NO_CHANGES: "No changes",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "Auto trading disabled by server",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "Auto trading disabled by client",
            mt5.TRADE_RETCODE_LOCKED: "Request locked",
            mt5.TRADE_RETCODE_FROZEN: "Order or position frozen",
            mt5.TRADE_RETCODE_INVALID_FILL: "Invalid fill",
            mt5.TRADE_RETCODE_CONNECTION: "No connection",
            mt5.TRADE_RETCODE_ONLY_REAL: "Only real accounts allowed",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Orders limit reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
            mt5.TRADE_RETCODE_INVALID_ORDER: "Incorrect or prohibited order type",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "Position with specified identifier already closed",
        }
        
        return error_dict.get(retcode, f"Unknown error code: {retcode}")
    
    def _log_successful_order(self, result: OrderResult, order_type: OrderType, 
                             volume: float, comment: str, is_recovery: bool):
        """Log successful order execution"""
        order_info = {
            "ticket": result.ticket,
            "type": order_type.name,
            "volume": volume,
            "price": result.price,
            "comment": comment,
            "is_recovery": is_recovery,
            "timestamp": datetime.now()
        }
        
        self.execution_history.append(order_info)
        self.logger.info(f"Order executed: {order_info}")
    
    def _log_failed_order(self, result: OrderResult, order_type: OrderType, 
                         volume: float, comment: str):
        """Log failed order execution"""
        self.logger.error(f"Order failed: {order_type.name} {volume} lots - {result.error_msg}")
    
    def _update_execution_time(self, execution_time: float):
        """Update average execution time"""
        if self.successful_orders == 1:
            self.average_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_execution_time = (alpha * execution_time + 
                                         (1 - alpha) * self.average_execution_time)
    
    def get_execution_stats(self) -> Dict:
        """Get execution performance statistics"""
        success_rate = (self.successful_orders / self.total_orders * 100) if self.total_orders > 0 else 0
        
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": round(success_rate, 2),
            "average_execution_time": round(self.average_execution_time, 3),
            "execution_mode": self.execution_mode.value,
            "max_slippage": self.max_slippage,
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }