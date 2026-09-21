import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="AI Real-Time Intraday Analyzer", layout="wide")

st.title("🤖 AI Real-Time Intraday Analyzer")
st.caption("VWAP + EMA 20/50 + Liquidity + Market Structure + Candlestick Confirmation")

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol", "^NSEI")
    interval = st.selectbox("Candle Timeframe", ["1m", "2m", "5m", "15m"], index=2)
    period = st.selectbox("Data Window", ["1d", "5d", "1mo"], index=1)
    capital = st.number_input("Your Capital (₹)", min_value=0.0, value=10000.0, step=1000.0)
    risk_pct = st.number_input("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    rr = st.number_input("Target Risk:Reward", 0.5, 10.0, 2.0, 0.5)
    min_score = st.slider("Minimum Signal Score", 50, 100, 70)
    refresh = st.button("🔄 Refresh")

@st.cache_data(ttl=10)
def get_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    cols = ["Open","High","Low","Close","Volume"]
    return df[[c for c in cols if c in df.columns]].dropna()

df = get_data(symbol, interval, period)

if df.empty:
    st.error("Data nahi mila. Symbol ya timeframe check karein.")
    st.stop()

# Indicators
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

typical = (df["High"] + df["Low"] + df["Close"]) / 3
dates = pd.Series(df.index.date, index=df.index)
cum_pv = (typical * df["Volume"]).groupby(dates).cumsum()
cum_vol = df["Volume"].groupby(dates).cumsum().replace(0, np.nan)
df["VWAP"] = cum_pv / cum_vol

tr = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - df["Close"].shift()).abs(),
    (df["Low"] - df["Close"].shift()).abs()
], axis=1).max(axis=1)
df["ATR"] = tr.rolling(14).mean()
df["VolMA20"] = df["Volume"].rolling(20).mean()

# Candlestick patterns
po, pc = df["Open"].shift(), df["Close"].shift()
bull_prev = pc < po
bear_prev = pc > po
bull_now = df["Close"] > df["Open"]
bear_now = df["Close"] < df["Open"]

df["BullishEngulfing"] = bull_prev & bull_now & (df["Open"] <= pc) & (df["Close"] >= po)
df["BearishEngulfing"] = bear_prev & bear_now & (df["Open"] >= pc) & (df["Close"] <= po)

body = (df["Close"] - df["Open"]).abs()
rng = (df["High"] - df["Low"]).replace(0, np.nan)
upper = df["High"] - df[["Open","Close"]].max(axis=1)
lower = df[["Open","Close"]].min(axis=1) - df["Low"]

df["Hammer"] = (lower >= 2*body) & (upper <= body) & ((body/rng) < 0.4)
df["ShootingStar"] = (upper >= 2*body) & (lower <= body) & ((body/rng) < 0.4)

# Liquidity
df["Prev20High"] = df["High"].shift(1).rolling(20).max()
df["Prev20Low"] = df["Low"].shift(1).rolling(20).min()
df["BullSweep"] = (df["Low"] < df["Prev20Low"]) & (df["Close"] > df["Prev20Low"])
df["BearSweep"] = (df["High"] > df["Prev20High"]) & (df["Close"] < df["Prev20High"])

# Simple structure proxy
df["HH"] = df["High"] > df["High"].shift(1)
df["HL"] = df["Low"] > df["Low"].shift(1)
df["LH"] = df["High"] < df["High"].shift(1)
df["LL"] = df["Low"] < df["Low"].shift(1)

def analyze(row):
    bull = bear = 0
    br, sr = [], []

    if row["Close"] > row["VWAP"]:
        bull += 15; br.append("Price > VWAP")
    elif row["Close"] < row["VWAP"]:
        bear += 15; sr.append("Price < VWAP")

    if row["EMA20"] > row["EMA50"]:
        bull += 15; br.append("EMA20 > EMA50")
    elif row["EMA20"] < row["EMA50"]:
        bear += 15; sr.append("EMA20 < EMA50")

    if row["HH"] or row["HL"]:
        bull += 10; br.append("Bullish structure")
    if row["LH"] or row["LL"]:
        bear += 10; sr.append("Bearish structure")

    if row["BullSweep"]:
        bull += 20; br.append("Bullish liquidity sweep")
    if row["BearSweep"]:
        bear += 20; sr.append("Bearish liquidity sweep")

    if row["BullishEngulfing"] or row["Hammer"]:
        bull += 20; br.append("Bullish candle confirmation")
    if row["BearishEngulfing"] or row["ShootingStar"]:
        bear += 20; sr.append("Bearish candle confirmation")

    if pd.notna(row["VolMA20"]) and row["Volume"] > row["VolMA20"]:
        if bull >= bear:
            bull += 10; br.append("High volume")
        else:
            bear += 10; sr.append("High volume")

    if bull >= min_score and bull > bear:
        return "BUY", bull, br
    if bear >= min_score and bear > bull:
        return "SELL", bear, sr
    return "NO TRADE", max(bull, bear), br if bull >= bear else sr

last = df.iloc[-1]
signal, score, reasons = analyze(last)

price = float(last["Close"])
atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0
risk_money = capital * risk_pct / 100
sl_dist = atr * 1.5 if atr > 0 else 0

if signal == "BUY" and sl_dist:
    sl, target = price - sl_dist, price + sl_dist * rr
elif signal == "SELL" and sl_dist:
    sl, target = price + sl_dist, price - sl_dist * rr
else:
    sl = target = np.nan

qty = int(risk_money / sl_dist) if sl_dist > 0 else 0

# Dashboard
a,b,c,d,e = st.columns(5)
a.metric("Last Price", f"₹{price:,.2f}")
b.metric("AI Signal", signal)
c.metric("Score", f"{score}/100")
d.metric("VWAP", f"₹{float(last['VWAP']):,.2f}")
e.metric("EMA 20 / 50", f"{float(last['EMA20']):.2f} / {float(last['EMA50']):.2f}")

st.subheader("🧠 AI Analysis")
st.write("**Confirmation:** " + (", ".join(reasons) if reasons else "No strong confirmation"))

if signal != "NO TRADE":
    p,q,r,s = st.columns(4)
    p.metric("Entry", f"₹{price:,.2f}")
    q.metric("Stop Loss", f"₹{sl:,.2f}")
    r.metric("Target", f"₹{target:,.2f}")
    s.metric("Position Qty", str(qty))

st.subheader("🕯️ Candlestick Chart")
plot_df = df.tail(150)
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"], name="Price"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["VWAP"], name="VWAP"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA20"], name="EMA20"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA50"], name="EMA50"))
fig.update_layout(height=650, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔎 Latest Candle")
latest = pd.DataFrame([{
    "Time": str(df.index[-1]),
    "Open": float(last["Open"]), "High": float(last["High"]),
    "Low": float(last["Low"]), "Close": float(last["Close"]),
    "Volume": float(last["Volume"]),
    "Bullish Engulfing": bool(last["BullishEngulfing"]),
    "Bearish Engulfing": bool(last["BearishEngulfing"]),
    "Hammer": bool(last["Hammer"]),
    "Shooting Star": bool(last["ShootingStar"]),
    "Bull Sweep": bool(last["BullSweep"]),
    "Bear Sweep": bool(last["BearSweep"])
}])
st.dataframe(latest, use_container_width=True)

st.subheader("⚠️ Important")
st.warning(
    "Ye educational/paper-analysis prototype hai. Yahoo Finance feed exchange-grade execution feed nahi hai. "
    "Real automated trading ke liye broker WebSocket/API, order-status handling, slippage, fees, "
    "kill-switch aur compliance controls add karne honge. AI score future profit guarantee nahi karta."
)
