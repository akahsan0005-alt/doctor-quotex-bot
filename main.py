import yfinance as yf
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from sklearn.linear_model import LogisticRegression
import os

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

bot = Bot(token=TOKEN)
updater = Updater(token=TOKEN, use_context=True)
dispatcher = updater.dispatcher

# =========================
# BOT STATE
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

# =========================
# HELPER FUNCTIONS
# =========================
def is_authorized(update):
    return str(update.effective_chat.id) == CHAT_ID

def otc_time_allowed():
    minute = datetime.utcnow().minute
    return minute <= 20

# =========================
# AUTO AI RETRAIN
# =========================
def auto_retrain():
    global ai_model, LAST_RETRAIN
    if LAST_RETRAIN and (datetime.utcnow() - LAST_RETRAIN).days < AUTO_RETRAIN_INTERVAL:
        return
    frames = []
    for asset_name, ASSET in ASSETS.items():
        df = yf.download(ASSET, interval=TIMEFRAME, period=LOOKBACK)
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
    data = pd.concat(frames)
    data.dropna(inplace=True)
    X = data[["rsi", "ema_dist", "body_ratio"]]
    y = data["label"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, AI_MODEL_PATH)
    ai_model = joblib.load(AI_MODEL_PATH)
    LAST_RETRAIN = datetime.utcnow()
    bot.send_message(chat_id=CHAT_ID, text=f"🧠 AI retrained successfully at {LAST_RETRAIN}")

# =========================
# AI FILTER
# =========================
def ai_allows_trade(last):
    if not AI_ENABLED:
        return True
    features = np.array([[
        last["rsi"],
        abs(last["ema50"] - last["ema200"]),
        abs(last["Close"] - last["Open"]) / (last["High"] - last["Low"])
    ]])
    prob = ai_model.predict_proba(features)[0][1]
    if last["rsi"] < 45 or last["rsi"] > 60:
        return prob > CONFIDENCE_RISKY
    else:
        return prob > CONFIDENCE_NORMAL

# =========================
# SIGNAL LOGIC
# =========================
def generate_signal(asset_name, ASSET):
    if not otc_time_allowed():
        return "SKIP ⏸ (OTC Time Filter)"
    df = yf.download(ASSET, interval=TIMEFRAME, period=LOOKBACK)
    df.dropna(inplace=True)
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    df["rsi"] = RSIIndicator(df["Close"], 14).rsi()
    last = df.iloc[-1]
    body = abs(last["Close"] - last["Open"])
    candle_range = last["High"] - last["Low"]
    if candle_range == 0: return "SKIP ⏸"
    upper_wick = last["High"] - max(last["Open"], last["Close"])
    lower_wick = min(last["Open"], last["Close"]) - last["Low"]
    if body / candle_range < 0.6: return "SKIP ⏸ (Weak Candle)"
    ema_distance = abs(last["ema50"] - last["ema200"])
    recent_range = df.tail(20)["High"].max() - df.tail(20)["Low"].min()
    if ema_distance < recent_range * 0.15: return "SKIP ⏸ (EMA Compression)"
    recent = df.tail(10)
    weak = sum(1 for _, row in recent.iterrows() if (abs(row["Close"] - row["Open"]) / (row["High"] - row["Low"]) if row["High"] != row["Low"] else 0) < 0.4)
    if weak >= 7: return "SKIP ⏸ (Ranging)"
    sr = df.tail(30)
    support = sr["Low"].min()
    resistance = sr["High"].max()
    buffer = (resistance - support) * 0.1
    near_support = last["Low"] <= support + buffer
    near_resistance = last["High"] >= resistance - buffer
    if near_support and last["ema50"] > last["ema200"] and 40 < last["rsi"] < 55 and lower_wick > body * 1.5:
        if not ai_allows_trade(last): return "SKIP ⏸ (AI Filter)"
        return f"CALL 📈 (AI + PA) - {asset_name}"
    if near_resistance and last["ema50"] < last["ema200"] and 45 < last["rsi"] < 60 and upper_wick > body * 1.5:
        if not ai_allows_trade(last): return "SKIP ⏸ (AI Filter)"
        return f"PUT 📉 (AI + PA) - {asset_name}"
    return "SKIP ⏸"

# =========================
# TELEGRAM COMMANDS
# =========================
def start_bot(update, context): 
    global BOT_ACTIVE; BOT_ACTIVE=True
    if is_authorized(update): update.message.reply_text("✅ Bot STARTED")

def stop_bot(update, context): 
    global BOT_ACTIVE; BOT_ACTIVE=False
    if is_authorized(update): update.message.reply_text("⛔ Bot STOPPED")

def pause_bot(update, context): 
    global BOT_ACTIVE; BOT_ACTIVE=False
    if is_authorized(update): update.message.reply_text("⏸ Bot PAUSED")

def ai_on(update, context): 
    global AI_ENABLED; AI_ENABLED=True
    if is_authorized(update): update.message.reply_text("🧠 AI Filter ENABLED")

def ai_off(update, context): 
    global AI_ENABLED; AI_ENABLED=False
    if is_authorized(update): update.message.reply_text("⚠️ AI Filter DISABLED")

def set_confidence(update, context): 
    global CONFIDENCE_NORMAL, CONFIDENCE_RISKY
    if not is_authorized(update): return
    try: 
        normal=float(context.args[0]); risky=float(context.args[1])
        CONFIDENCE_NORMAL=normal; CONFIDENCE_RISKY=risky
        update.message.reply_text(f"🎯 Confidence updated\nNormal: {normal}\nRisky: {risky}")
    except: 
        update.message.reply_text("Usage: /confidence 0.58 0.65")

def status(update, context): 
    if not is_authorized(update): return
    rate=0 if TOTAL_TRADES==0 else (WINS/TOTAL_TRADES)*100
    msg=f"\n📊 BOT STATUS\nActive: {BOT_ACTIVE}\nAI Filter: {AI_ENABLED}\nConfidence Normal: {CONFIDENCE_NORMAL}\nConfidence Risky: {CONFIDENCE_RISKY}\nTotal Trades: {TOTAL_TRADES}\nWins: {WINS}\nLosses: {LOSSES}\nWin Rate: {rate:.2f}%\n"
    update.message.reply_text(msg)

def win(update, context): 
    global TOTAL_TRADES,WINS,DAILY_LOG; TOTAL_TRADES+=1; WINS+=1; DAILY_LOG.append("WIN")
    if is_authorized(update): update.message.reply_text("✅ Win recorded")

def loss(update, context): 
    global TOTAL_TRADES,LOSSES,DAILY_LOG; TOTAL_TRADES+=1; LOSSES+=1; DAILY_LOG.append("LOSS")
    if is_authorized(update): update.message.reply_text("❌ Loss recorded")

def winrate(update, context): 
    if not is_authorized(update): return
    rate=0 if TOTAL_TRADES==0 else (WINS/TOTAL_TRADES)*100
    update.message.reply_text(f"\n📈 LIVE PERFORMANCE\nTrades: {TOTAL_TRADES}\nWins: {WINS}\nLosses: {LOSSES}\nWin Rate: {rate:.2f}%\n")

# Register commands
dispatcher.add_handler(CommandHandler("startbot", start_bot))
dispatcher.add_handler(CommandHandler("stopbot", stop_bot))
dispatcher.add_handler(CommandHandler("pause", pause_bot))
dispatcher.add_handler(CommandHandler("aion", ai_on))
dispatcher.add_handler(CommandHandler("aioff", ai_off))
dispatcher.add_handler(CommandHandler("confidence", set_confidence))
dispatcher.add_handler(CommandHandler("status", status))
dispatcher.add_handler(CommandHandler("win", win))
dispatcher.add_handler(CommandHandler("loss", loss))
dispatcher.add_handler(CommandHandler("winrate", winrate))
updater.start_polling()

# =========================
# DAILY REPORT
# =========================
def daily_report():
    global WINS, LOSSES, TOTAL_TRADES, DAILY_LOG
    rate=0 if TOTAL_TRADES==0 else (WINS/TOTAL_TRADES)*100
    msg=f"\n📊 DAILY REPORT (UTC)\nTotal Trades: {TOTAL_TRADES}\nWins: {WINS}\nLosses: {LOSSES}\nWin Rate: {rate:.2f}%\nDiscipline > Frequency"
    bot.send_message(chat_id=CHAT_ID, text=msg)
    TOTAL_TRADES=WINS=LOSSES=0
    DAILY_LOG=[]

# =========================
# MAIN LOOP 24/7
# =========================
while True:
    try:
        auto_retrain()  # retrain AI if interval passed

        # Daily report
        global LAST_REPORT_DAY
        today=datetime.utcnow().date()
        if LAST_REPORT_DAY!=today:
            if LAST_REPORT_DAY is not None: daily_report()
            LAST_REPORT_DAY=today

        if not BOT_ACTIVE:
            time.sleep(10)
            continue

        for name, symbol in ASSETS.items():
            signal = generate_signal(name, symbol)
            if "CALL" in signal or "PUT" in signal:
                LAST_SIGNAL={"asset":name,"time":datetime.utcnow(),"signal":signal}
                msg=f"\n📊 QUOTEX AI SIGNAL\nAsset: {name}\nTimeframe: 1 Minute\nSignal: {signal}\nReply with:\n/win ✅  /loss ❌\n"
                bot.send_message(chat_id=CHAT_ID,text=msg)
                break  # Only one trade per candle
        time.sleep(60)
    except Exception as e:
        print("Error:", e)
        time.sleep(60)
