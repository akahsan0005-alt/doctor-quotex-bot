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
from telegram.ext import Updater, CommandHandler
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
TOKEN = os.environ["TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

ASSETS = {
    "EURUSD OTC": "EURUSD=X",
    "GBPUSD OTC": "GBPUSD=X",
    "USDJPY OTC": "JPY=X"
}

TIMEFRAME = "1m"
LOOKBACK = "7d"
AUTO_RETRAIN_INTERVAL = 7  # days

# =========================
# TELEGRAM
# =========================
bot = Bot(token=TOKEN)
updater = Updater(token=TOKEN, use_context=True)
dispatcher = updater.dispatcher

# =========================
# GLOBAL STATE
# =========================
BOT_ACTIVE = True
AI_ENABLED = True

CONFIDENCE_NORMAL = 0.58
CONFIDENCE_RISKY = 0.65

TOTAL_TRADES = 0
WINS = 0
LOSSES = 0
DAILY_LOG = []

LAST_SIGNAL = None
LAST_REPORT_DAY = None
LAST_RETRAIN = None

AI_MODEL_PATH = "ai_filter.pkl"
ai_model = None

# =========================
# HELPERS
# =========================
def is_authorized(update):
    return str(update.effective_chat.id) == CHAT_ID

def otc_time_allowed():
    return datetime.utcnow().minute <= 20

# =========================
# AI RETRAIN
# =========================
def auto_retrain():
    global ai_model, LAST_RETRAIN

    if LAST_RETRAIN and (datetime.utcnow() - LAST_RETRAIN).days < AUTO_RETRAIN_INTERVAL:
        return

    frames = []
    for _, symbol in ASSETS.items():
        df = yf.download(symbol, interval=TIMEFRAME, period=LOOKBACK)
        df.dropna(inplace=True)

        df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
        df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
        df["rsi"] = RSIIndicator(df["Close"], 14).rsi()
        df["body"] = abs(df["Close"] - df["Open"])
        df["range"] = df["High"] - df["Low"]
        df["ema_dist"] = abs(df["ema50"] - df["ema200"])
        df["body_ratio"] = df["body"] / df["range"]
        df["future"] = df["Close"].shift(-1)
        df["label"] = (df["future"] > df["Close"]).astype(int)

        frames.append(df)

    data = pd.concat(frames).dropna()
    X = data[["rsi", "ema_dist", "body_ratio"]]
    y = data["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, AI_MODEL_PATH)
    ai_model = model
    LAST_RETRAIN = datetime.utcnow()

    bot.send_message(chat_id=CHAT_ID, text="🤖 AI retrained successfully")

# =========================
# AI FILTER
# =========================
def ai_allows_trade(last):
    if not AI_ENABLED or ai_model is None:
        return True

    features = np.array([[
        last["rsi"],
        abs(last["ema50"] - last["ema200"]),
        abs(last["Close"] - last["Open"]) / (last["High"] - last["Low"])
    ]])

    prob = ai_model.predict_proba(features)[0][1]

    if last["rsi"] < 45 or last["rsi"] > 60:
        return prob > CONFIDENCE_RISKY
    return prob > CONFIDENCE_NORMAL

# =========================
# SIGNAL LOGIC
# =========================
def generate_signal(name, symbol):
    if not otc_time_allowed():
        return None

    df = yf.download(symbol, interval=TIMEFRAME, period=LOOKBACK)
    df.dropna(inplace=True)

    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    df["rsi"] = RSIIndicator(df["Close"], 14).rsi()

    last = df.iloc[-1]
    body = abs(last["Close"] - last["Open"])
    rng = last["High"] - last["Low"]

    if rng == 0 or body / rng < 0.6:
        return None

    support = df.tail(30)["Low"].min()
    resistance = df.tail(30)["High"].max()
    buffer = (resistance - support) * 0.1

    if (
        last["Low"] <= support + buffer
        and last["ema50"] > last["ema200"]
        and 40 < last["rsi"] < 55
        and ai_allows_trade(last)
    ):
        return f"CALL - {name}"

    if (
        last["High"] >= resistance - buffer
        and last["ema50"] < last["ema200"]
        and 45 < last["rsi"] < 60
        and ai_allows_trade(last)
    ):
        return f"PUT - {name}"

    return None

# =========================
# COMMANDS
# =========================
def start_bot(update, context):
    global BOT_ACTIVE
    BOT_ACTIVE = True
    update.message.reply_text("✅ Bot started")

def stop_bot(update, context):
    global BOT_ACTIVE
    BOT_ACTIVE = False
    update.message.reply_text("⛔ Bot stopped")

def win(update, context):
    global TOTAL_TRADES, WINS
    TOTAL_TRADES += 1
    WINS += 1
    update.message.reply_text("✅ Win recorded")

def loss(update, context):
    global TOTAL_TRADES, LOSSES
    TOTAL_TRADES += 1
    LOSSES += 1
    update.message.reply_text("❌ Loss recorded")

dispatcher.add_handler(CommandHandler("startbot", start_bot))
dispatcher.add_handler(CommandHandler("stopbot", stop_bot))
dispatcher.add_handler(CommandHandler("win", win))
dispatcher.add_handler(CommandHandler("loss", loss))

updater.start_polling()

# =========================
# DAILY REPORT
# =========================
def daily_report():
    global TOTAL_TRADES, WINS, LOSSES

    rate = 0 if TOTAL_TRADES == 0 else (WINS / TOTAL_TRADES) * 100
    msg = (
        f"📊 DAILY REPORT\n"
        f"Trades: {TOTAL_TRADES}\n"
        f"Wins: {WINS}\n"
        f"Losses: {LOSSES}\n"
        f"Win Rate: {rate:.2f}%"
    )
    bot.send_message(chat_id=CHAT_ID, text=msg)

    TOTAL_TRADES = WINS = LOSSES = 0

# =========================
# MAIN LOOP
# =========================
while True:
    try:
        auto_retrain()

        today = datetime.utcnow().date()
        if LAST_REPORT_DAY != today:
            if LAST_REPORT_DAY is not None:
                daily_report()
            LAST_REPORT_DAY = today

        if BOT_ACTIVE:
            for name, symbol in ASSETS.items():
                signal = generate_signal(name, symbol)
                if signal:
                    LAST_SIGNAL = signal
                    bot.send_message(
                        chat_id=CHAT_ID,
                        text=f"📈 QUOTEX SIGNAL\n{signal}\nReply: /win or /loss"
                    )
                    break

        time.sleep(60)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(60)
