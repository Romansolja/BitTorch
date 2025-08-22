import asyncio
import websockets
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

console = Console()


class BTCWebSocketClient:
    def __init__(self, uri="ws://localhost:8000/ws"):
        self.uri = uri
        self.current_price = 0
        self.change_24h = 0
        self.high_24h = 0
        self.low_24h = 0
        self.volume = 0
        self.last_update = "Never"
        self.connection_status = "Disconnected"
        self.message_count = 0

    def create_display(self):
        """Create rich display layout"""
        layout = Layout()

        # Create status panel
        status_color = "green" if self.connection_status == "Connected" else "red"
        status_panel = Panel(
            f"[{status_color}]● {self.connection_status}[/{status_color}]\n"
            f"Messages: {self.message_count}\n"
            f"Last Update: {self.last_update}",
            title="Connection Status"
        )

        # Create price panel
        price_color = "green" if self.change_24h >= 0 else "red"
        price_panel = Panel(
            f"[bold white]${self.current_price:,.2f}[/bold white]\n"
            f"[{price_color}]{'+' if self.change_24h >= 0 else ''}{self.change_24h:.2f}%[/{price_color}]",
            title="Bitcoin Price"
        )

        # Create metrics table
        table = Table(title="24h Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("High", f"${self.high_24h:,.0f}")
        table.add_row("Low", f"${self.low_24h:,.0f}")
        table.add_row("Volume", f"{self.volume:,.0f}")

        # Combine layouts
        layout.split_column(
            Layout(status_panel, size=5),
            Layout(price_panel, size=5),
            Layout(table)
        )

        return layout

    async def connect(self):
        """Connect to WebSocket and listen for updates"""
        console.print("[yellow]Connecting to BitTorch WebSocket...[/yellow]")

        try:
            async with websockets.connect(self.uri) as websocket:
                self.connection_status = "Connected"
                console.print("[green]Connected successfully![/green]")

                # Send initial ping
                await websocket.send(json.dumps({"command": "ping"}))

                # Start live display
                with Live(self.create_display(), refresh_per_second=1) as live:
                    # Create ping task
                    async def ping_task():
                        while True:
                            await asyncio.sleep(30)
                            await websocket.send(json.dumps({"command": "ping"}))

                    ping = asyncio.create_task(ping_task())

                    try:
                        async for message in websocket:
                            data = json.loads(message)
                            self.handle_message(data)
                            live.update(self.create_display())
                    finally:
                        ping.cancel()

        except websockets.exceptions.WebSocketException as e:
            console.print(f"[red]WebSocket error: {e}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Disconnecting...[/yellow]")
        finally:
            self.connection_status = "Disconnected"
            console.print("[red]Disconnected[/red]")

    def handle_message(self, data):
        """Process incoming WebSocket message"""
        self.message_count += 1

        if data.get("type") in ["price_update", "initial"]:
            price_data = data.get("data", data)
            self.current_price = price_data.get("price", 0)
            self.change_24h = price_data.get("change_24h_percent", 0)
            self.high_24h = price_data.get("high_24h", 0)
            self.low_24h = price_data.get("low_24h", 0)
            self.volume = price_data.get("volume_24h", 0)
            self.last_update = datetime.now().strftime("%H:%M:%S")

            # Log significant changes
            if self.message_count > 1:  # Skip first message
                console.print(f"[cyan]Price updated: ${self.current_price:,.2f}[/cyan]")

        elif data.get("type") == "pong":
            console.print("[dim]Pong received[/dim]")

        elif data.get("type") == "error":
            console.print(f"[red]Error: {data.get('message')}[/red]")


async def main():
    """Main function with menu"""
    console.print(Panel.fit(
        "[bold cyan]BitTorch WebSocket Test Client[/bold cyan]\n"
        "Real-time Bitcoin price monitoring",
        border_style="cyan"
    ))

    # Menu
    console.print("\n[yellow]Options:[/yellow]")
    console.print("1. Connect to default WebSocket (localhost:8000)")
    console.print("2. Connect to custom URL")
    console.print("3. Exit")

    choice = console.input("\n[cyan]Select option: [/cyan]")

    if choice == "1":
        client = BTCWebSocketClient()
        await client.connect()
    elif choice == "2":
        url = console.input("[cyan]Enter WebSocket URL: [/cyan]")
        client = BTCWebSocketClient(url)
        await client.connect()
    elif choice == "3":
        console.print("[yellow]Goodbye![/yellow]")
    else:
        console.print("[red]Invalid option[/red]")


if __name__ == "__main__":
    # Install rich if not installed: pip install rich
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")