import yfinance as yf
import pandas as pd
import numpy as np
import time
import joblib
import os
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from telegram import Bot
from sklearn.linear_model import LogisticRegression

# ================= CONFIG =================
TOKEN = os.environ["TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

bot = Bot(token=TOKEN)

ASSETS = {
    "EURUSD OTC": "EURUSD=X",
    "GBPUSD OTC": "GBPUSD=X",
    "USDJPY OTC": "JPY=X"
}

TIMEFRAME = "1m"
LOOKBACK = "7d"

AI_MODEL_PATH = "ai_filter.pkl"
AUTO_RETRAIN_DAYS = 7

BOT_ACTIVE = True
TOTAL_TRADES = WINS = LOSSES = 0
LAST_RETRAIN = None
LAST_REPORT_DAY = None

# ================= AI =================
def retrain_ai():
    global LAST_RETRAIN

    if LAST_RETRAIN and (datetime.utcnow() - LAST_RETRAIN).days < AUTO_RETRAIN_DAYS:
        return

    frames = []
    for symbol in ASSETS.values():
        df = yf.download(symbol, interval=TIMEFRAME, period=LOOKBACK)
        df.dropna(inplace=True)

        df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
        df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
        df["rsi"] = RSIIndicator(df["Close"], 14).rsi()
        df["future"] = df["Close"].shift(-1)
        df["label"] = (df["future"] > df["Close"]).astype(int)

        frames.append(df)

    data = pd.concat(frames).dropna()

    X = data[["rsi", "ema50", "ema200"]]
    y = data["label"].values.ravel()  # ✅ FIXED

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, AI_MODEL_PATH)

    LAST_RETRAIN = datetime.utcnow()
    bot.send_message(chat_id=CHAT_ID, text="🤖 AI retrained")

# ================= SIGNAL =================
def generate_signal(name, symbol):
    df = yf.download(symbol, interval=TIMEFRAME, period=LOOKBACK)
    df.dropna(inplace=True)

    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    df["rsi"] = RSIIndicator(df["Close"], 14).rsi()

    last = df.iloc[-1]

    if last["ema50"] > last["ema200"] and 40 < last["rsi"] < 55:
        return f"📈 CALL {name}"

    if last["ema50"] < last["ema200"] and 45 < last["rsi"] < 60:
        return f"📉 PUT {name}"

    return None

# ================= MAIN LOOP =================
bot.send_message(chat_id=CHAT_ID, text="✅ Bot started")

while True:
    try:
        retrain_ai()

        today = datetime.utcnow().date()
        if LAST_REPORT_DAY != today:
            if LAST_REPORT_DAY:
                rate = 0 if TOTAL_TRADES == 0 else (WINS / TOTAL_TRADES) * 100
                bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"📊 Daily Report\nTrades: {TOTAL_TRADES}\nWin rate: {rate:.2f}%"
                )
                TOTAL_TRADES = WINS = LOSSES = 0
            LAST_REPORT_DAY = today

        if BOT_ACTIVE:
            for name, symbol in ASSETS.items():
                signal = generate_signal(name, symbol)
                if signal:
                    bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"{signal}\nReply with: /win or /loss"
                    )
                    break

        time.sleep(60)

    except Exception as e:
        bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Error: {e}")
        time.sleep(60)
