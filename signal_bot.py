import logging
import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from datetime import datetime

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
TOKEN = "8966564343:AAFi-V6EgY70Tjmqz26H456kibOH7smjgFY"

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True})

# Quotex popular assets mapped to data sources
QUOTEX_ASSETS = {
    # Crypto (Binance)
    "BTC/USD":  ("binance", "BTC/USDT"),
    "ETH/USD":  ("binance", "ETH/USDT"),
    "BNB/USD":  ("binance", "BNB/USDT"),
    "SOL/USD":  ("binance", "SOL/USDT"),
    "XRP/USD":  ("binance", "XRP/USDT"),
    "DOGE/USD": ("binance", "DOGE/USDT"),
    "ADA/USD":  ("binance", "ADA/USDT"),
    "LTC/USD":  ("binance", "LTC/USDT"),
    # Forex (Yahoo Finance)
    "EUR/USD":  ("yahoo", "EURUSD=X"),
    "GBP/USD":  ("yahoo", "GBPUSD=X"),
    "USD/JPY":  ("yahoo", "JPY=X"),
    "AUD/USD":  ("yahoo", "AUDUSD=X"),
    "USD/CAD":  ("yahoo", "CAD=X"),
    "EUR/GBP":  ("yahoo", "EURGBP=X"),
    # Commodities (Yahoo Finance)
    "XAU/USD":  ("yahoo", "GC=F"),   # Gold
    "XAG/USD":  ("yahoo", "SI=F"),   # Silver
    "OIL/USD":  ("yahoo", "CL=F"),   # Crude Oil
    "NAS100":   ("yahoo", "NQ=F"),   # Nasdaq
    "SP500":    ("yahoo", "ES=F"),   # S&P 500
}

TIMEFRAMES_BINANCE = {
    "1m  ⚡": "1m",
    "5m  🔥": "5m",
    "15m 📈": "15m",
    "1h  🕐": "1h",
    "4h  📊": "4h",
}

TIMEFRAMES_YAHOO = {
    "1m  ⚡": ("1m",  "1d"),
    "5m  🔥": ("5m",  "5d"),
    "15m 📈": ("15m", "5d"),
    "1h  🕐": ("1h",  "1mo"),
    "4h  📊": ("4h",  "3mo"),
}

# ──────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def compute_bollinger(series, period=20):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + 2*std, mid, mid - 2*std

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_stoch_rsi(series, period=14):
    rsi   = compute_rsi(series, period)
    min_r = rsi.rolling(period).min()
    max_r = rsi.rolling(period).max()
    denom = (max_r - min_r).replace(0, np.nan)
    return (rsi - min_r) / denom * 100

# ──────────────────────────────────────────────
#  DATA FETCH
# ──────────────────────────────────────────────

async def fetch_binance(symbol: str, timeframe: str, limit=150):
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception as e:
        logger.error(f"Binance error: {e}")
        return None

def fetch_yahoo(ticker: str, interval: str, period: str):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        # Flatten MultiIndex if present
        if hasattr(df.columns, 'levels'):
            df.columns = ['_'.join(filter(None, map(str, c))).strip('_') for c in df.columns]
        df = df.rename(columns={
            "datetime": "ts", "date": "ts",
            "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "vol"
        })
        # Pick close column
        close_cols = [c for c in df.columns if 'close' in c.lower()]
        if close_cols:
            df['close'] = df[close_cols[0]]
        return df[["ts","close"]].dropna()
    except Exception as e:
        logger.error(f"Yahoo error: {e}")
        return None

# ──────────────────────────────────────────────
#  RESOLVE MARKET
# ──────────────────────────────────────────────

async def resolve_market(raw: str):
    """Return (source, symbol/ticker, display_name) or None"""
    raw = raw.upper().strip().replace(" ", "").replace("-", "/")

    # Check Quotex asset map first
    for display, (source, sym) in QUOTEX_ASSETS.items():
        if raw == display.replace("/", "").replace(" ", "") or raw == display:
            return source, sym, display

    # Try Binance directly
    candidates = []
    if "/" in raw:
        candidates.append(raw)
    else:
        for quote in ["USDT", "USD", "BTC", "ETH"]:
            if raw.endswith(quote):
                candidates.append(f"{raw[:-len(quote)]}/{quote}")
            else:
                candidates.append(f"{raw}/{quote}")

    try:
        markets = await exchange.load_markets()
        for sym in candidates:
            if sym in markets:
                return "binance", sym, sym
    except:
        pass

    return None

# ──────────────────────────────────────────────
#  SIGNAL ENGINE
# ──────────────────────────────────────────────

def analyse(close_series: pd.Series) -> dict:
    close = close_series.reset_index(drop=True)

    rsi                  = compute_rsi(close)
    macd, msig, mhist    = compute_macd(close)
    bb_up, bb_mid, bb_lo = compute_bollinger(close)
    ema9  = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50)
    stoch = compute_stoch_rsi(close)

    r   = rsi.iloc[-1]
    mh  = mhist.iloc[-1]
    mh1 = mhist.iloc[-2]
    c   = close.iloc[-1]
    e9  = ema9.iloc[-1]
    e21 = ema21.iloc[-1]
    e50 = ema50.iloc[-1]
    bup = bb_up.iloc[-1]
    blo = bb_lo.iloc[-1]
    st  = stoch.iloc[-1]

    score = 0
    reasons = []

    if r < 30:
        score += 2; reasons.append("🟢 RSI Oversold (<30)")
    elif r > 70:
        score -= 2; reasons.append("🔴 RSI Overbought (>70)")
    elif r < 45:
        score += 1; reasons.append("🟡 RSI mildly bearish zone")
    elif r > 55:
        score -= 1; reasons.append("🟡 RSI mildly bullish zone")

    if mh > 0 and mh1 <= 0:
        score += 2; reasons.append("🟢 MACD Bullish Crossover")
    elif mh < 0 and mh1 >= 0:
        score -= 2; reasons.append("🔴 MACD Bearish Crossover")
    elif mh > 0:
        score += 1; reasons.append("🟡 MACD Histogram Positive")
    else:
        score -= 1; reasons.append("🟡 MACD Histogram Negative")

    if e9 > e21 > e50:
        score += 2; reasons.append("🟢 EMA Uptrend (9>21>50)")
    elif e9 < e21 < e50:
        score -= 2; reasons.append("🔴 EMA Downtrend (9<21<50)")
    elif e9 > e21:
        score += 1; reasons.append("🟡 Short EMA Bullish")
    else:
        score -= 1; reasons.append("🟡 Short EMA Bearish")

    if c <= blo:
        score += 2; reasons.append("🟢 Price at Lower Bollinger Band")
    elif c >= bup:
        score -= 2; reasons.append("🔴 Price at Upper Bollinger Band")

    if st < 20:
        score += 1; reasons.append("🟢 Stoch RSI Oversold")
    elif st > 80:
        score -= 1; reasons.append("🔴 Stoch RSI Overbought")

    strength = min(abs(score) / 10 * 100, 100)
    filled   = int(strength / 10)
    bar      = "█" * filled + "░" * (10 - filled)

    if score >= 5:
        signal, emoji = "STRONG BUY 🟢", "🚀"
    elif score >= 3:
        signal, emoji = "BUY 🟢", "📈"
    elif score >= 1:
        signal, emoji = "WEAK BUY 🟡", "🔼"
    elif score <= -5:
        signal, emoji = "STRONG SELL 🔴", "📉"
    elif score <= -3:
        signal, emoji = "SELL 🔴", "🔻"
    elif score <= -1:
        signal, emoji = "WEAK SELL 🟠", "⚠️"
    else:
        signal, emoji = "NEUTRAL ⚪", "😐"

    return {
        "signal": signal, "emoji": emoji,
        "score": score, "strength": strength, "bar": bar,
        "rsi": round(r, 2), "macd_h": round(mh, 6),
        "stoch": round(st, 2),
        "ema9": round(e9, 5), "ema21": round(e21, 5), "ema50": round(e50, 5),
        "price": round(c, 5), "reasons": reasons,
    }

# ──────────────────────────────────────────────
#  KEYBOARDS
# ──────────────────────────────────────────────

def tf_keyboard(source: str, sym: str, display: str):
    rows = []
    tfs  = TIMEFRAMES_BINANCE if source == "binance" else TIMEFRAMES_YAHOO
    for label in tfs:
        cb = f"tf:{source}:{sym}:{label}:{display}"
        rows.append([InlineKeyboardButton(label, callback_data=cb)])
    rows.append([InlineKeyboardButton("🔄 Change Market", callback_data="change")])
    return InlineKeyboardMarkup(rows)

def quotex_menu_keyboard():
    rows = []
    items = list(QUOTEX_ASSETS.keys())
    for i in range(0, len(items), 2):
        row = [InlineKeyboardButton(items[i], callback_data=f"qx:{items[i]}")]
        if i+1 < len(items):
            row.append(InlineKeyboardButton(items[i+1], callback_data=f"qx:{items[i+1]}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("✏️ Type Custom Market", callback_data="change")])
    return InlineKeyboardMarkup(rows)

# ──────────────────────────────────────────────
#  HANDLERS
# ──────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "╔══════════════════════════╗\n"
        "║   📊  SIGNAL  BOT  PRO   ║\n"
        "╚══════════════════════════╝\n\n"
        "✅ Supports *Quotex* assets:\n"
        "• Crypto (BTC, ETH, SOL...)\n"
        "• Forex (EUR/USD, GBP/USD...)\n"
        "• Gold, Silver, Oil, Indices\n\n"
        "📌 *Choose from list or type any market!*\n\n"
        "⚠️ _Educational use only. Trade at your own risk._"
    )
    await update.message.reply_text(
        text, parse_mode="Markdown",
        reply_markup=quotex_menu_keyboard()
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    msg  = await update.message.reply_text("🔍 Looking up market...")

    result = await resolve_market(text)
    if not result:
        await msg.edit_text(
            f"❌ *'{text}'* not found.\n\n"
            "Try: `BTC`, `EURUSD`, `XAUUSD`, `SOL/USDT`",
            parse_mode="Markdown"
        )
        return

    source, sym, display = result
    await msg.edit_text(
        f"✅ Market: *{display}*\n\n👇 Choose timeframe:",
        parse_mode="Markdown",
        reply_markup=tf_keyboard(source, sym, display)
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data

    if data == "change":
        await query.edit_message_text(
            "✏️ *Type your market:*\n\n"
            "Examples:\n`BTC` `ETH` `EURUSD` `XAUUSD` `SOL/USDT`",
            parse_mode="Markdown"
        )
        return

    if data == "menu":
        await query.edit_message_text(
            "👇 *Choose a market:*",
            parse_mode="Markdown",
            reply_markup=quotex_menu_keyboard()
        )
        return

    if data.startswith("qx:"):
        display  = data.split("qx:")[1]
        source, sym = QUOTEX_ASSETS[display]
        await query.edit_message_text(
            f"✅ Market: *{display}*\n\n👇 Choose timeframe:",
            parse_mode="Markdown",
            reply_markup=tf_keyboard(source, sym, display)
        )
        return

    if data.startswith("tf:"):
        parts   = data.split(":", 4)
        source  = parts[1]
        sym     = parts[2]
        tf_label = parts[3]
        display = parts[4]

        await query.edit_message_text(
            f"⏳ Analysing *{display}* | `{tf_label.strip()}`...",
            parse_mode="Markdown"
        )

        # Fetch data
        df = None
        if source == "binance":
            tf = TIMEFRAMES_BINANCE[tf_label]
            df = await fetch_binance(sym, tf)
            if df is not None:
                close = df["close"]
        else:
            interval, period = TIMEFRAMES_YAHOO[tf_label]
            df = await asyncio.to_thread(fetch_yahoo, sym, interval, period)
            if df is not None:
                close = df["close"]

        if df is None or len(df) < 50:
            await query.edit_message_text(
                "❌ Data fetch failed. Try a different timeframe.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Back", callback_data="menu")
                ]])
            )
            return

        res = analyse(close)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        reason_text = "\n".join(res["reasons"]) if res["reasons"] else "—"

        msg_text = (
            f"╔══════════════════════════╗\n"
            f"║  {res['emoji']}  SIGNAL RESULT        ║\n"
            f"╚══════════════════════════╝\n\n"
            f"📌 *Market:*    `{display}`\n"
            f"⏱ *Timeframe:* `{tf_label.strip()}`\n"
            f"💰 *Price:*     `{res['price']}`\n"
            f"🕐 *Time:*      {now}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *Signal:*   {res['signal']}\n"
            f"💪 *Strength:* `{res['bar']}` {res['strength']:.0f}%\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Indicators:*\n"
            f"• RSI:        `{res['rsi']}`\n"
            f"• MACD Hist:  `{res['macd_h']}`\n"
            f"• Stoch RSI:  `{res['stoch']}`\n"
            f"• EMA 9:      `{res['ema9']}`\n"
            f"• EMA 21:     `{res['ema21']}`\n"
            f"• EMA 50:     `{res['ema50']}`\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 *Analysis:*\n{reason_text}\n\n"
            f"⚠️ _Educational use only. Trade at your own risk._"
        )

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"tf:{source}:{sym}:{tf_label}:{display}")],
            [InlineKeyboardButton("⏱ Change Timeframe", callback_data=f"qx:{display}" if display in QUOTEX_ASSETS else "change")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")],
        ])

        await query.edit_message_text(msg_text, parse_mode="Markdown", reply_markup=kb)

# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("✅ Bot started!")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
