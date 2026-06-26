import os
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
#  CONFIG — Token environment variable থেকে নেবে
# ──────────────────────────────────────────────
TOKEN = os.environ.get("TOKEN", "")
if not TOKEN:
    raise ValueError("TOKEN environment variable not set!")

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True})

QUOTEX_ASSETS = {
    "BTC/USD":  ("binance", "BTC/USDT"),
    "ETH/USD":  ("binance", "ETH/USDT"),
    "BNB/USD":  ("binance", "BNB/USDT"),
    "SOL/USD":  ("binance", "SOL/USDT"),
    "XRP/USD":  ("binance", "XRP/USDT"),
    "DOGE/USD": ("binance", "DOGE/USDT"),
    "ADA/USD":  ("binance", "ADA/USDT"),
    "LTC/USD":  ("binance", "LTC/USDT"),
    "EUR/USD":  ("yahoo", "EURUSD=X"),
    "GBP/USD":  ("yahoo", "GBPUSD=X"),
    "USD/JPY":  ("yahoo", "JPY=X"),
    "AUD/USD":  ("yahoo", "AUDUSD=X"),
    "USD/CAD":  ("yahoo", "CAD=X"),
    "EUR/GBP":  ("yahoo", "EURGBP=X"),
    "XAU/USD":  ("yahoo", "GC=F"),
    "XAG/USD":  ("yahoo", "SI=F"),
    "OIL/USD":  ("yahoo", "CL=F"),
    "NAS100":   ("yahoo", "NQ=F"),
    "SP500":    ("yahoo", "ES=F"),
}

TIMEFRAMES_BINANCE = {
    "1m ⚡": "1m",
    "5m 🔥": "5m",
    "15m 📈": "15m",
    "1h 🕐": "1h",
    "4h 📊": "4h",
}

TIMEFRAMES_YAHOO = {
    "1m ⚡":  ("1m",  "1d"),
    "5m 🔥":  ("5m",  "5d"),
    "15m 📈": ("15m", "5d"),
    "1h 🕐":  ("1h",  "1mo"),
    "4h 📊":  ("4h",  "3mo"),
}

# Expiry suggestion based on timeframe
EXPIRY_MAP = {
    "1m":  ("1-2 min",  "⚡"),
    "5m":  ("5 min",    "🔥"),
    "15m": ("15 min",   "📈"),
    "1h":  ("1 hour",   "🕐"),
    "4h":  ("4 hours",  "📊"),
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
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col if c]).strip('_') for col in df.columns]
        df.columns = [str(c).lower() for c in df.columns]
        # Find close column
        close_col = next((c for c in df.columns if 'close' in c), None)
        ts_col    = next((c for c in df.columns if c in ['datetime','date','index']), None)
        if not close_col:
            return None
        result = pd.DataFrame()
        result['ts']    = df[ts_col] if ts_col else df.index
        result['close'] = pd.to_numeric(df[close_col], errors='coerce')
        return result.dropna()
    except Exception as e:
        logger.error(f"Yahoo error: {e}")
        return None

# ──────────────────────────────────────────────
#  RESOLVE MARKET
# ──────────────────────────────────────────────

async def resolve_market(raw: str):
    raw = raw.upper().strip().replace(" ", "").replace("-", "/")

    for display, (source, sym) in QUOTEX_ASSETS.items():
        clean_display = display.replace("/", "").replace(" ", "")
        if raw == clean_display or raw == display:
            return source, sym, display

    candidates = []
    if "/" in raw:
        candidates.append(raw)
    else:
        for quote in ["USDT", "USD", "BTC", "ETH"]:
            if raw.endswith(quote) and len(raw) > len(quote):
                candidates.append(f"{raw[:-len(quote)]}/{quote}")
            else:
                candidates.append(f"{raw}/{quote}")

    try:
        markets = await exchange.load_markets()
        for sym in candidates:
            if sym in markets:
                return "binance", sym, sym
    except Exception as e:
        logger.error(f"Market resolve error: {e}")

    return None

# ──────────────────────────────────────────────
#  SIGNAL ENGINE
# ──────────────────────────────────────────────

def analyse(close_series: pd.Series, tf_key: str = "5m") -> dict:
    close = close_series.reset_index(drop=True).astype(float)

    rsi                  = compute_rsi(close)
    macd, msig, mhist    = compute_macd(close)
    bb_up, bb_mid, bb_lo = compute_bollinger(close)
    ema9  = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50)
    stoch = compute_stoch_rsi(close)

    r   = float(rsi.iloc[-1])
    mh  = float(mhist.iloc[-1])
    mh1 = float(mhist.iloc[-2])
    c   = float(close.iloc[-1])
    e9  = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e50 = float(ema50.iloc[-1])
    bup = float(bb_up.iloc[-1])
    blo = float(bb_lo.iloc[-1])
    st  = float(stoch.iloc[-1])

    score = 0
    reasons = []

    if r < 30:
        score += 2; reasons.append("🟢 RSI Oversold (<30)")
    elif r > 70:
        score -= 2; reasons.append("🔴 RSI Overbought (>70)")
    elif r < 45:
        score += 1; reasons.append("🟡 RSI mildly bearish")
    elif r > 55:
        score -= 1; reasons.append("🟡 RSI mildly bullish")

    if mh > 0 and mh1 <= 0:
        score += 2; reasons.append("🟢 MACD Bullish Crossover")
    elif mh < 0 and mh1 >= 0:
        score -= 2; reasons.append("🔴 MACD Bearish Crossover")
    elif mh > 0:
        score += 1; reasons.append("🟡 MACD Positive")
    else:
        score -= 1; reasons.append("🟡 MACD Negative")

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
        signal, direction, dir_emoji = "STRONG BUY", "⬆️ UP", "🚀"
    elif score >= 3:
        signal, direction, dir_emoji = "BUY", "⬆️ UP", "📈"
    elif score >= 1:
        signal, direction, dir_emoji = "WEAK BUY", "⬆️ UP", "🔼"
    elif score <= -5:
        signal, direction, dir_emoji = "STRONG SELL", "⬇️ DOWN", "📉"
    elif score <= -3:
        signal, direction, dir_emoji = "SELL", "⬇️ DOWN", "🔻"
    elif score <= -1:
        signal, direction, dir_emoji = "WEAK SELL", "⬇️ DOWN", "⚠️"
    else:
        signal, direction, dir_emoji = "NEUTRAL", "↔️ WAIT", "😐"

    expiry, exp_emoji = EXPIRY_MAP.get(tf_key, ("5 min", "🔥"))

    return {
        "signal": signal, "direction": direction,
        "dir_emoji": dir_emoji, "expiry": expiry, "exp_emoji": exp_emoji,
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
    await update.message.reply_text(
        "╔══════════════════════════╗\n"
        "║   📊  SIGNAL  BOT  PRO   ║\n"
        "╚══════════════════════════╝\n\n"
        "✅ Supports *Quotex* assets:\n"
        "• Crypto (BTC, ETH, SOL...)\n"
        "• Forex (EUR/USD, GBP/USD...)\n"
        "• Gold, Silver, Oil, Indices\n\n"
        "📌 *Choose from list or type any market!*\n\n"
        "⚠️ _Educational use only. Trade at your own risk._",
        parse_mode="Markdown",
        reply_markup=quotex_menu_keyboard()
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    msg  = await update.message.reply_text("🔍 Looking up market...")

    result = await resolve_market(text)
    if not result:
        await msg.edit_text(
            f"❌ *'{text}'* not found.\n\nTry: `BTC`, `EURUSD`, `XAUUSD`, `SOL`",
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
            "✏️ *Type your market:*\n\nExamples:\n`BTC` `ETH` `EURUSD` `XAUUSD` `SOL`",
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
        display = data[3:]
        if display not in QUOTEX_ASSETS:
            await query.edit_message_text("❌ Market not found.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data="menu")]]))
            return
        source, sym = QUOTEX_ASSETS[display]
        await query.edit_message_text(
            f"✅ Market: *{display}*\n\n👇 Choose timeframe:",
            parse_mode="Markdown",
            reply_markup=tf_keyboard(source, sym, display)
        )
        return

    if data.startswith("tf:"):
        parts    = data.split(":", 4)
        source   = parts[1]
        sym      = parts[2]
        tf_label = parts[3]
        display  = parts[4]

        await query.edit_message_text(
            f"⏳ Analysing *{display}* | `{tf_label.strip()}`...",
            parse_mode="Markdown"
        )

        df    = None
        close = None

        if source == "binance":
            tf = TIMEFRAMES_BINANCE[tf_label]
            df = await fetch_binance(sym, tf)
            if df is not None:
                close  = df["close"]
                tf_key = tf
        else:
            interval, period = TIMEFRAMES_YAHOO[tf_label]
            df = await asyncio.to_thread(fetch_yahoo, sym, interval, period)
            if df is not None:
                close  = df["close"]
                tf_key = interval

        if df is None or close is None or len(df) < 50:
            await query.edit_message_text(
                "❌ Data fetch failed. Try a different timeframe.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="menu")]])
            )
            return

        res = analyse(close, tf_key)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        reason_text = "\n".join(res["reasons"]) if res["reasons"] else "—"

        msg_text = (
            f"╔══════════════════════════╗\n"
            f"║ {res['dir_emoji']}  SIGNAL RESULT         ║\n"
            f"╚══════════════════════════╝\n\n"
            f"📌 *Market:*    `{display}`\n"
            f"⏱ *Timeframe:* `{tf_label.strip()}`\n"
            f"💰 *Price:*     `{res['price']}`\n"
            f"🕐 *Time:*      {now}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *Signal:*    *{res['signal']}*\n"
            f"📌 *Direction:* *{res['direction']}*\n"
            f"⏳ *Expiry:*    *{res['exp_emoji']} {res['expiry']}*\n"
            f"💪 *Strength:*  `{res['bar']}` {res['strength']:.0f}%\n\n"
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
            [InlineKeyboardButton("🔄 Refresh Signal", callback_data=f"tf:{source}:{sym}:{tf_label}:{display}")],
            [InlineKeyboardButton("⏱ Change Timeframe", callback_data=f"qx:{display}" if display in QUOTEX_ASSETS else f"ctf:{source}:{sym}:{display}")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")],
        ])

        await query.edit_message_text(msg_text, parse_mode="Markdown", reply_markup=kb)
        return

    if data.startswith("ctf:"):
        parts   = data.split(":", 3)
        source  = parts[1]
        sym     = parts[2]
        display = parts[3]
        await query.edit_message_text(
            f"✅ Market: *{display}*\n\n👇 Choose timeframe:",
            parse_mode="Markdown",
            reply_markup=tf_keyboard(source, sym, display)
        )

# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("✅ Bot started!")
    await app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
