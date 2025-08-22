from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, List
import asyncio
import yfinance as yf
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        # Store active connections
        self.active_connections: Set[WebSocket] = set()
        self.latest_price_data: Dict = {}
        self.price_history: List[Dict] = []  # Keep last 100 price points
        self.max_history = 100

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

        # Send current data immediately if available
        if self.latest_price_data:
            await websocket.send_json({
                "type": "initial",
                "data": self.latest_price_data,
                "history": self.price_history[-20:]  # Send last 20 points
            })

    async def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except:
            await self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        # Store latest data
        self.latest_price_data = message.get("data", {})

        # Add to history if it's a price update
        if message.get("type") == "price_update":
            self.price_history.append(self.latest_price_data)
            if len(self.price_history) > self.max_history:
                self.price_history.pop(0)

        # Send to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)


class PriceFetcher:
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.running = False
        self.fetch_interval = 30  # seconds
        self.quick_fetch_interval = 5  # seconds for rapid updates
        self.use_rapid_mode = False

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
            except Exception as e:
                logger.error(f"Price fetcher error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def fetch_and_broadcast(self):
        """Fetch current BTC price and broadcast to clients"""
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

            # Broadcast to all clients
            await self.manager.broadcast({
                "type": "price_update",
                "data": data
            })

            logger.info(f"Broadcasted price: ${current_price:,.2f} to {len(self.manager.active_connections)} clients")

        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            # Send error notification to clients
            await self.manager.broadcast({
                "type": "error",
                "message": "Failed to fetch latest price",
                "timestamp": datetime.now().isoformat()
            })

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

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON received: {message}")
    except Exception as e:
        logger.error(f"Error handling message: {e}")