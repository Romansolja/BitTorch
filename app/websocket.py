from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, List, Optional, Any
import asyncio
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
import numpy as np
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TimeFrame(str, Enum):
    """Supported timeframes for chart data"""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"


class ConnectionManager:
    def __init__(self):
        # Store active connections
        self.active_connections: Set[WebSocket] = set()
        self.latest_price_data: Dict = {}
        self.price_history: List[Dict] = []  # Keep last 100 price points
        self.max_history = 100

        # Chart subscriptions: {websocket: {"symbol": str, "interval": str}}
        self.chart_subscriptions: Dict[WebSocket, Dict] = {}

        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, connection_type: str = "price"):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Store connection metadata
        self.connection_metadata[websocket] = {
            "type": connection_type,
            "connected_at": datetime.now(),
            "last_ping": datetime.now()
        }

        logger.info(f"Client connected ({connection_type}). Total connections: {len(self.active_connections)}")

        # Send current data immediately if available
        if connection_type == "price" and self.latest_price_data:
            await websocket.send_json({
                "type": "initial",
                "data": self.latest_price_data,
                "history": self.price_history[-20:]  # Send last 20 points
            })

    async def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        self.active_connections.discard(websocket)
        self.chart_subscriptions.pop(websocket, None)
        self.connection_metadata.pop(websocket, None)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
            self.connection_metadata[websocket]["last_ping"] = datetime.now()
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)

    async def broadcast(self, message: dict, connection_type: str = "price"):
        """Broadcast message to all connected clients of specific type"""
        if not self.active_connections:
            return

        # Store latest data for price updates
        if message.get("type") == "price_update":
            self.latest_price_data = message.get("data", {})
            self.price_history.append(self.latest_price_data)
            if len(self.price_history) > self.max_history:
                self.price_history.pop(0)

        # Send to connections based on type
        disconnected = set()
        for connection in self.active_connections:
            # Check connection type
            conn_meta = self.connection_metadata.get(connection, {})
            if conn_meta.get("type") != connection_type and connection_type != "all":
                continue

            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)

    async def subscribe_to_chart(self, websocket: WebSocket, symbol: str, interval: str):
        """Subscribe websocket to chart data for specific symbol and interval"""
        self.chart_subscriptions[websocket] = {
            "symbol": symbol,
            "interval": interval,
            "subscribed_at": datetime.now()
        }
        logger.info(f"Client subscribed to {symbol} {interval} chart data")


class ChartDataManager:
    """Manages historical chart data streaming"""

    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.running = False
        self.update_intervals = {
            TimeFrame.ONE_MIN: 60,
            TimeFrame.FIVE_MIN: 300,
            TimeFrame.FIFTEEN_MIN: 900,
            TimeFrame.ONE_HOUR: 3600,
            TimeFrame.ONE_DAY: 86400
        }
        # Cache for chart data to reduce API calls
        self.chart_cache: Dict[str, Dict] = {}
        self.cache_ttl = 30  # seconds

    async def start(self):
        """Start chart data streaming"""
        self.running = True
        logger.info("Starting chart data manager...")

        # Create tasks for different update intervals
        tasks = []
        for interval in [TimeFrame.ONE_MIN, TimeFrame.FIVE_MIN]:
            task = asyncio.create_task(self.update_loop(interval))
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def update_loop(self, interval: TimeFrame):
        """Update loop for specific timeframe"""
        update_seconds = self.update_intervals[interval]

        while self.running:
            try:
                # Get all unique symbols subscribed for this interval
                symbols = set()
                for sub in self.manager.chart_subscriptions.values():
                    if sub.get("interval") == interval.value:
                        symbols.add(sub.get("symbol"))

                # Fetch and broadcast data for each symbol
                for symbol in symbols:
                    await self.fetch_and_broadcast_chart(symbol, interval)

                await asyncio.sleep(update_seconds)

            except Exception as e:
                logger.error(f"Chart update loop error ({interval}): {e}")
                await asyncio.sleep(60)

    async def fetch_and_broadcast_chart(self, symbol: str, interval: TimeFrame):
        """Fetch and broadcast chart data for specific symbol and interval"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval.value}"
            cached = self.chart_cache.get(cache_key)

            if cached and (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl:
                chart_data = cached["data"]
            else:
                # Fetch fresh data
                chart_data = await self.fetch_chart_data(symbol, interval)
                self.chart_cache[cache_key] = {
                    "data": chart_data,
                    "timestamp": datetime.now()
                }

            # Send to subscribed clients
            for websocket, subscription in self.manager.chart_subscriptions.items():
                if subscription.get("symbol") == symbol and subscription.get("interval") == interval.value:
                    await self.manager.send_personal_message({
                        "type": "chart_update",
                        "symbol": symbol,
                        "interval": interval.value,
                        "data": chart_data
                    }, websocket)

        except Exception as e:
            logger.error(f"Error fetching chart data for {symbol} {interval}: {e}")

    async def fetch_chart_data(self, symbol: str, interval: TimeFrame) -> Dict:
        """Fetch OHLCV data for charting"""
        try:
            ticker = yf.Ticker(symbol)

            # Determine period based on interval
            period_map = {
                TimeFrame.ONE_MIN: "1d",
                TimeFrame.FIVE_MIN: "5d",
                TimeFrame.FIFTEEN_MIN: "5d",
                TimeFrame.ONE_HOUR: "1mo",
                TimeFrame.ONE_DAY: "6mo"
            }

            period = period_map.get(interval, "1mo")

            # Fetch historical data
            history = ticker.history(period=period, interval=interval.value)

            if history.empty:
                return {"candles": [], "indicators": {}}

            # Convert to candlestick format
            candles = []
            for idx, row in history.iterrows():
                candles.append({
                    "time": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"])
                })

            # Calculate technical indicators
            closes = history["Close"].values
            indicators = self.calculate_indicators(closes)

            return {
                "candles": candles[-100:],  # Last 100 candles
                "indicators": indicators,
                "summary": {
                    "current": float(closes[-1]),
                    "change_24h": float((closes[-1] / closes[-min(24, len(closes) - 1)] - 1) * 100) if len(
                        closes) > 1 else 0,
                    "high_24h": float(history["High"].tail(24).max()) if len(history) >= 24 else float(
                        history["High"].max()),
                    "low_24h": float(history["Low"].tail(24).min()) if len(history) >= 24 else float(
                        history["Low"].min()),
                    "volume_24h": float(history["Volume"].tail(24).sum()) if len(history) >= 24 else float(
                        history["Volume"].sum())
                }
            }

        except Exception as e:
            logger.error(f"Error fetching chart data: {e}")
            return {"candles": [], "indicators": {}}

    def calculate_indicators(self, prices: np.ndarray) -> Dict:
        """Calculate technical indicators for chart"""
        if len(prices) < 2:
            return {}

        indicators = {}

        try:
            # Moving Averages
            if len(prices) >= 7:
                indicators["sma_7"] = float(np.mean(prices[-7:]))
            if len(prices) >= 25:
                indicators["sma_25"] = float(np.mean(prices[-25:]))
            if len(prices) >= 99:
                indicators["sma_99"] = float(np.mean(prices[-99:]))

            # RSI (14 period)
            if len(prices) >= 15:
                indicators["rsi"] = self.calculate_rsi(prices, 14)

            # Bollinger Bands (20 period)
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                std_20 = np.std(prices[-20:])
                indicators["bb_upper"] = float(sma_20 + (2 * std_20))
                indicators["bb_middle"] = float(sma_20)
                indicators["bb_lower"] = float(sma_20 - (2 * std_20))

            # MACD
            if len(prices) >= 26:
                exp1 = self.ema(prices, 12)
                exp2 = self.ema(prices, 26)
                macd = exp1 - exp2
                signal = self.ema(macd, 9)
                indicators["macd"] = float(macd[-1])
                indicators["macd_signal"] = float(signal[-1])
                indicators["macd_histogram"] = float(macd[-1] - signal[-1])

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def ema(self, values: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(values, weights, mode='full')[:len(values)]
        ema[:period] = ema[period]
        return ema

    async def stop(self):
        """Stop chart data manager"""
        self.running = False
        logger.info("Chart data manager stopped")


class PriceFetcher:
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.running = False
        self.fetch_interval = 30  # seconds
        self.quick_fetch_interval = 5  # seconds for rapid updates
        self.use_rapid_mode = False
        self.retry_count = 0
        self.max_retries = 3

    async def start(self):
        """Start the price fetching loop"""
        self.running = True
        logger.info("Starting price fetcher...")

        # Initial fetch
        await self.fetch_and_broadcast()

        # Continue fetching
        while self.running:
            try:
                interval = self.quick_fetch_interval if self.use_rapid_mode else self.fetch_interval
                await asyncio.sleep(interval)
                await self.fetch_and_broadcast()
                self.retry_count = 0  # Reset on success
            except Exception as e:
                logger.error(f"Price fetcher error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def fetch_and_broadcast(self):
        """Fetch current BTC price and broadcast to clients"""
        for attempt in range(self.max_retries):
            try:
                # Get BTC ticker
                btc = yf.Ticker("BTC-USD")
                info = btc.info

                # Get current price and 24h data
                history = btc.history(period="1d", interval="1m")
                current_price = history['Close'].iloc[-1] if not history.empty else info.get('regularMarketPrice', 0)

                # Calculate 24h change
                history_24h = btc.history(period="1d", interval="1h")
                if not history_24h.empty and len(history_24h) > 1:
                    price_24h_ago = history_24h['Close'].iloc[0]
                    change_24h = current_price - price_24h_ago
                    change_24h_percent = (change_24h / price_24h_ago) * 100
                else:
                    change_24h = 0
                    change_24h_percent = 0

                # Get additional market data
                data = {
                    "price": float(current_price),
                    "timestamp": datetime.now().isoformat(),
                    "change_24h": float(change_24h),
                    "change_24h_percent": float(change_24h_percent),
                    "volume_24h": info.get('volume24Hr', 0),
                    "market_cap": info.get('marketCap', 0),
                    "high_24h": float(history_24h['High'].max()) if not history_24h.empty else 0,
                    "low_24h": float(history_24h['Low'].min()) if not history_24h.empty else 0,
                }

                # Broadcast to all price clients
                await self.manager.broadcast({
                    "type": "price_update",
                    "data": data
                }, connection_type="price")

                logger.info(f"Broadcasted price: ${current_price:,.2f} to price subscribers")
                return  # Success, exit retry loop

            except Exception as e:
                self.retry_count = attempt + 1
                logger.error(f"Error fetching price (attempt {self.retry_count}/{self.max_retries}): {e}")

                if self.retry_count >= self.max_retries:
                    # Send error notification to clients
                    await self.manager.broadcast({
                        "type": "error",
                        "message": f"Failed to fetch price after {self.max_retries} attempts",
                        "timestamp": datetime.now().isoformat()
                    }, connection_type="price")
                else:
                    await asyncio.sleep(2)  # Short delay before retry

    async def stop(self):
        """Stop the price fetcher"""
        self.running = False
        logger.info("Price fetcher stopped")

    def set_rapid_mode(self, enabled: bool):
        """Enable/disable rapid price updates"""
        self.use_rapid_mode = enabled
        logger.info(f"Rapid mode {'enabled' if enabled else 'disabled'}")


# Singleton instances
manager = ConnectionManager()
price_fetcher = PriceFetcher(manager)
chart_manager = ChartDataManager(manager)


# WebSocket message handler for client commands
async def handle_client_message(websocket: WebSocket, message: str):
    """Handle incoming messages from clients"""
    try:
        data = json.loads(message)
        command = data.get("command")

        if command == "ping":
            # Respond to ping
            await manager.send_personal_message({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }, websocket)

        elif command == "get_history":
            # Send price history
            await manager.send_personal_message({
                "type": "history",
                "data": manager.price_history
            }, websocket)

        elif command == "rapid_mode":
            # Toggle rapid updates (admin only - add auth check in production)
            enabled = data.get("enabled", False)
            price_fetcher.set_rapid_mode(enabled)
            await manager.send_personal_message({
                "type": "rapid_mode",
                "enabled": enabled
            }, websocket)

        elif command == "subscribe_chart":
            # Subscribe to chart data
            symbol = data.get("symbol", "BTC-USD")
            interval = data.get("interval", "1m")

            # Validate interval
            if interval in [tf.value for tf in TimeFrame]:
                await manager.subscribe_to_chart(websocket, symbol, interval)

                # Send initial chart data
                chart_data = await chart_manager.fetch_chart_data(symbol, TimeFrame(interval))
                await manager.send_personal_message({
                    "type": "chart_initial",
                    "symbol": symbol,
                    "interval": interval,
                    "data": chart_data
                }, websocket)
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Invalid interval: {interval}"
                }, websocket)

        elif command == "unsubscribe_chart":
            # Unsubscribe from chart data
            manager.chart_subscriptions.pop(websocket, None)
            await manager.send_personal_message({
                "type": "chart_unsubscribed",
                "timestamp": datetime.now().isoformat()
            }, websocket)

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON received: {message}")
    except Exception as e:
        logger.error(f"Error handling message: {e}")


# Heartbeat task to keep connections alive
async def heartbeat_task():
    """Send periodic heartbeat to all connections"""
    while True:
        await asyncio.sleep(45)  # Every 45 seconds
        await manager.broadcast({
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat()
        }, connection_type="all")