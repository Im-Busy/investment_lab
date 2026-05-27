"""P28-21: MCP stock data server for AI agents.

Model Context Protocol server exposing stock data and technical analysis
to AI coding assistants (Cursor, Claude, Gemini). Provides tools for
price data, technical indicators, and pattern detection — making the
project's analysis capabilities available through MCP.

Source: stock-sdk MCP server pattern.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MCP_AVAILABLE = False
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
    )

    _MCP_AVAILABLE = True
except ImportError:
    logger.info("mcp package not installed — MCP server requires 'pip install mcp'")


class StockDataTools:
    """Stock data tools callable by MCP clients."""

    @staticmethod
    def get_price(
        symbol: str,
        start: str,
        end: str | None = None,
        period: str = "1y",
    ) -> dict:
        """Fetch OHLCV data and return summary stats."""
        end = end or datetime.now().strftime("%Y-%m-%d")
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                return {"error": f"No data for {symbol} {start}→{end}"}

            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            return {
                "symbol": symbol,
                "start": start,
                "end": end,
                "bars": len(df),
                "latest_price": float(close.iloc[-1]) if len(close) > 0 else None,
                "latest_date": str(df.index[-1])[:10],
                "return_total_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2)
                if len(close) > 0
                else None,
                "high_52w": float(close.max()),
                "low_52w": float(close.min()),
                "ma_50": float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None,
                "ma_200": float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_technicals(
        symbol: str,
        period: str = "6mo",
    ) -> dict:
        """Compute technical indicators: RSI, MACD, ATR, BB, ADX."""
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                return {"error": f"No data for {symbol}"}

            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]

            latest: dict[str, float | str | None] = {
                "symbol": symbol,
                "date": str(df.index[-1])[:10],
            }

            if len(close) >= 15:
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = (-delta).clip(lower=0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                latest["rsi_14"] = (
                    round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else None
                )

                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9).mean()
                macd_hist = macd_line - signal_line
                latest["macd_line"] = round(float(macd_line.iloc[-1]), 4)
                latest["macd_signal"] = round(float(signal_line.iloc[-1]), 4)
                latest["macd_histogram"] = round(float(macd_hist.iloc[-1]), 4)

                tr = pd.concat(
                    [
                        high - low,
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = tr.rolling(14).mean()
                latest["atr_14"] = (
                    round(float(atr.iloc[-1]), 4) if not pd.isna(atr.iloc[-1]) else None
                )

                sma20 = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                bb_upper = sma20 + 2 * std20
                bb_lower = sma20 - 2 * std20
                latest["bb_upper"] = round(float(bb_upper.iloc[-1]), 2)
                latest["bb_middle"] = round(float(sma20.iloc[-1]), 2)
                latest["bb_lower"] = round(float(bb_lower.iloc[-1]), 2)
                latest["bb_position"] = (
                    round(
                        float(
                            (close.iloc[-1] - bb_lower.iloc[-1])
                            / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                            * 100
                        ),
                        1,
                    )
                    if bb_upper.iloc[-1] != bb_lower.iloc[-1]
                    else None
                )

                plus_dm = high.diff().clip(lower=0)
                minus_dm = (-low.diff()).clip(lower=0)
                atr14 = atr
                plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
                minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(14).mean()
                latest["adx_14"] = (
                    round(float(adx.iloc[-1]), 1) if not pd.isna(adx.iloc[-1]) else None
                )

            return latest
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_multi_symbol(
        symbols: str,
        start: str,
        end: str | None = None,
    ) -> dict:
        """Fetch and compare multiple symbols."""
        end = end or datetime.now().strftime("%Y-%m-%d")
        symbol_list = [s.strip() for s in symbols.split(",")]
        results = {}
        for sym in symbol_list[:8]:
            results[sym] = StockDataTools.get_price(sym, start, end)
        return {"symbols": results, "count": len(results)}


def _create_mcp_server() -> Server:
    """Create and configure the MCP server with registered tools."""
    server = Server("investment-stock-server")
    data_tools = StockDataTools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="stock_get_price",
                description="Get price data and summary stats for a stock symbol. Returns latest price, returns, 52w high/low, MA50/200.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Ticker symbol (e.g., SPY, AAPL, BTC-USD)",
                        },
                        "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end": {
                            "type": "string",
                            "description": "End date YYYY-MM-DD (default: today)",
                        },
                        "period": {
                            "type": "string",
                            "description": "Alternative to start/end: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max",
                        },
                    },
                    "required": ["symbol", "start"],
                },
            ),
            Tool(
                name="stock_get_technicals",
                description="Get technical indicators for a stock: RSI(14), MACD, ATR(14), Bollinger Bands, ADX(14).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol"},
                        "period": {
                            "type": "string",
                            "description": "Data period: 1mo,3mo,6mo,1y,2y (default: 6mo)",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="stock_get_multi",
                description="Compare multiple stock symbols side-by-side. Up to 8 symbols.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "string",
                            "description": "Comma-separated tickers (e.g., SPY,QQQ,XLK)",
                        },
                        "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end": {
                            "type": "string",
                            "description": "End date YYYY-MM-DD (default: today)",
                        },
                    },
                    "required": ["symbols", "start"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "stock_get_price":
            result = data_tools.get_price(
                symbol=arguments["symbol"],
                start=arguments["start"],
                end=arguments.get("end"),
                period=arguments.get("period", "1y"),
            )
        elif name == "stock_get_technicals":
            result = data_tools.get_technicals(
                symbol=arguments["symbol"],
                period=arguments.get("period", "6mo"),
            )
        elif name == "stock_get_multi":
            result = data_tools.get_multi(
                symbols=arguments["symbols"],
                start=arguments["start"],
                end=arguments.get("end"),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


def run_mcp_server() -> None:
    """Start the MCP stock data server on stdio."""
    if not _MCP_AVAILABLE:
        raise ImportError("MCP server requires 'pip install mcp'. Install with: uv add mcp")

    import asyncio

    server = _create_mcp_server()

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    run_mcp_server()
