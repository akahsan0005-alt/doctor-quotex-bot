import os
import pandas as pd
import pandas_ta as ta
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOL = "OTC"
TIMEFRAME = "1m"

# =========================
# PRICE DATA (PLACEHOLDER)
# =========================
def get_candles():
    """
    Replace this function with:
    - TradingView webhook
    - Your own candle feed
    - CSV upload
    """
    df = pd.read_csv("candles.csv")  # columns: time, open, high, low, close
    return df.tail(100)

# =========================
# STRATEGY ENGINE
# =========================
def analyze_market(df):
    df["ema9"] = ta.ema(df["close"], length=9)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    stoch = ta.stochrsi(df["close"])
    df["atr"] = ta.atr(df["high"], df["low"], df["close"])

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if (
        last.ema9 > last.ema21
        and 52 < last.rsi < 68
        and macd.iloc[-1]["MACDh_12_26_9"] > 0
        and stoch.iloc[-1]["STOCHRSIk_14_14_3_3"] > stoch.iloc[-2]["STOCHRSIk_14_14_3_3"]
        and last.atr > df["atr"].mean()
        and last.close > prev.close
    ):
        return "CALL"

    if (
        last.ema9 < last.ema21
        and 32 < last.rsi < 48
        and macd.iloc[-1]["MACDh_12_26_9"] < 0
        and stoch.iloc[-1]["STOCHRSIk_14_14_3_3"] < stoch.iloc[-2]["STOCHRSIk_14_14_3_3"]
        and last.atr > df["atr"].mean()
        and last.close < prev.close
    ):
        return "PUT"

    return None

# =========================
# TELEGRAM COMMAND
# =========================
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = get_candles()
        direction = analyze_market(df)

        if direction:
            msg = f"""
📊 *QUOTEX OTC SIGNAL*
⏱ Timeframe: 1 Minute
📈 Direction: *{direction}*
🚫 Martingale: NO
🕒 Signal Time: {datetime.utcnow()} UTC
"""
        else:
            msg = "❌ No high-probability setup found. Waiting for confirmation."

        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")

# =========================
# BOT START
# =========================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("signal", signal))
    app.run_polling()

if __name__ == "__main__":
    main()
