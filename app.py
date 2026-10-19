import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Crypto Paper Trader", page_icon="📈", layout="wide")

st.title("📈 AI Crypto Paper Trader")
st.caption("Paper trading only • Public market data • No exchange keys • No real orders")

with st.sidebar:
    st.header("Bot settings")
    exchange_name = st.selectbox("Market data source", ["Binance", "Kraken", "Coinbase"], index=1)
    symbol = st.selectbox("Trading pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BTC/USD", "ETH/USD"], index=0)
    timeframe = st.selectbox("Candlestick timeframe", ["5m", "15m", "1h", "4h"], index=1)
    starting_cash = st.number_input("Starting paper cash (USD)", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=500.0)
    risk_pct = st.slider("Cash used per buy", min_value=1, max_value=25, value=10, step=1) / 100
    threshold = st.slider("Minimum model confidence to trade", min_value=0.50, max_value=0.80, value=0.58, step=0.01)
    fee_pct = st.slider("Simulated fee per trade (%)", min_value=0.0, max_value=1.0, value=0.10, step=0.01) / 100
    refresh_seconds = st.selectbox("Refresh interval", [15, 30, 60, 120], index=2)
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh dashboard", value=False)
    run_bot = st.checkbox("Allow bot to simulate trades", value=False)
    if st.button("Reset paper account", use_container_width=True):
        for key in ["cash", "position", "trades", "equity_history", "last_candle"]:
            st.session_state.pop(key, None)
        st.rerun()

if "cash" not in st.session_state:
    st.session_state.cash = float(starting_cash)
    st.session_state.position = 0.0
    st.session_state.trades = []
    st.session_state.equity_history = []
    st.session_state.last_candle = None
if "position" not in st.session_state:
    st.session_state.position = 0.0
if "trades" not in st.session_state:
    st.session_state.trades = []
if "equity_history" not in st.session_state:
    st.session_state.equity_history = []

def make_exchange(name):
    exchanges = {
        "Binance": ccxt.binance,
        "Kraken": ccxt.kraken,
        "Coinbase": ccxt.coinbase,
    }
    return exchanges[name]({"enableRateLimit": True, "timeout": 15000})

@st.cache_data(ttl=20, show_spinner=False)
def get_ohlcv(exchange_name, symbol, timeframe):
    exchange = make_exchange(exchange_name)
    # Coinbase/Kraken may not support USDT pairs. Surface a clear error rather than silently substituting.
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
    if not rows:
        raise RuntimeError("The exchange returned no candles for this pair.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def build_features(df):
    x = df.copy()
    x["ret_1"] = x["close"].pct_change()
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_6"] = x["close"].pct_change(6)
    x["volatility"] = x["ret_1"].rolling(10).std()
    x["sma_ratio_5"] = x["close"] / x["close"].rolling(5).mean() - 1
    x["sma_ratio_20"] = x["close"] / x["close"].rolling(20).mean() - 1
    x["range_pct"] = (x["high"] - x["low"]) / x["close"]
    x["volume_change"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["target"] = (x["close"].shift(-1) > x["close"]).astype(int)
    return x

def train_and_predict(df):
    features = ["ret_1", "ret_3", "ret_6", "volatility", "sma_ratio_5",
                "sma_ratio_20", "range_pct", "volume_change", "rsi"]
    data = build_features(df)
    train = data.iloc[:-1].dropna(subset=features + ["target"])
    latest = data.iloc[[-1]][features]
    if len(train) < 60 or latest[features].isna().any(axis=None):
        return None, None, "Not enough clean candle data to train the model."
    # A small Random Forest is retrained on recent candles. This is a demo model, not a validated edge.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=120, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=42
        ))
    ])
    model.fit(train[features], train["target"])
    probabilities = model.predict_proba(latest[features])[0]
    classes = list(model.named_steps["rf"].classes_)
    up_prob = float(probabilities[classes.index(1)]) if 1 in classes else 0.5
    return up_prob, model, None

def fmt_money(value):
    return f"${value:,.2f}"

try:
    with st.spinner("Fetching public market candles and training the model..."):
        df = get_ohlcv(exchange_name, symbol, timeframe)
        current_price = float(df["close"].iloc[-1])
        up_prob, model, model_error = train_and_predict(df)
except Exception as exc:
    st.error(
        f"Could not fetch market data for **{symbol}** from **{exchange_name}**. "
        "Check your internet connection and whether that pair is supported by the selected exchange."
    )
    st.code(str(exc))
    st.info("Try Binance with BTC/USDT or Kraken/Coinbase with a pair listed by that exchange.")
    st.stop()

if model_error:
    st.warning(model_error)
    st.stop()

down_prob = 1 - up_prob
if up_prob >= threshold:
    signal = "BUY"
elif down_prob >= threshold:
    signal = "SELL"
else:
    signal = "HOLD"

position_value = st.session_state.position * current_price
equity = st.session_state.cash + position_value
initial = float(starting_cash)
pnl = equity - initial
pnl_pct = pnl / initial * 100 if initial else 0

# Simulated execution happens at most once per newly closed candle.
latest_candle_id = str(df["timestamp"].iloc[-1])
trade_note = None
if run_bot and st.session_state.last_candle != latest_candle_id:
    if signal == "BUY" and st.session_state.cash > 10:
        spend = st.session_state.cash * risk_pct
        fee = spend * fee_pct
        qty = max(0.0, (spend - fee) / current_price)
        if qty > 0:
            st.session_state.cash -= spend
            st.session_state.position += qty
            trade = {
                "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "Side": "BUY", "Price": current_price, "Quantity": qty,
                "Notional": spend, "Fee": fee, "Reason": f"Model up probability {up_prob:.1%}"
            }
            st.session_state.trades.insert(0, trade)
            trade_note = f"Simulated BUY: {qty:.6f} {symbol.split('/')[0]} at {fmt_money(current_price)}"
    elif signal == "SELL" and st.session_state.position > 0:
        qty = st.session_state.position
        gross = qty * current_price
        fee = gross * fee_pct
        proceeds = gross - fee
        st.session_state.cash += proceeds
        st.session_state.position = 0.0
        trade = {
            "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Side": "SELL", "Price": current_price, "Quantity": qty,
            "Notional": gross, "Fee": fee, "Reason": f"Model down probability {down_prob:.1%}"
        }
        st.session_state.trades.insert(0, trade)
        trade_note = f"Simulated SELL: {qty:.6f} {symbol.split('/')[0]} at {fmt_money(current_price)}"
    st.session_state.last_candle = latest_candle_id

position_value = st.session_state.position * current_price
equity = st.session_state.cash + position_value
pnl = equity - initial
pnl_pct = pnl / initial * 100 if initial else 0
st.session_state.equity_history.append({
    "time": datetime.now(timezone.utc),
    "equity": equity,
    "cash": st.session_state.cash,
    "position_value": position_value
})
st.session_state.equity_history = st.session_state.equity_history[-500:]

if trade_note:
    st.success(trade_note)

# Header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest candle close", fmt_money(current_price))
c2.metric("Paper equity", fmt_money(equity), f"{pnl_pct:+.2f}% vs starting cash")
c3.metric("Cash available", fmt_money(st.session_state.cash))
c4.metric("Open position", f"{st.session_state.position:.6f} {symbol.split('/')[0]}")

left, right = st.columns([1.7, 1])
with left:
    st.subheader(f"{symbol} price")
    fig = go.Figure(data=[go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=symbol
    )])
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Model signal")
    color = "green" if signal == "BUY" else ("red" if signal == "SELL" else "gray")
    st.markdown(f"### :{color}[{signal}]")
    st.metric("Model probability: next candle UP", f"{up_prob:.1%}")
    st.metric("Model probability: next candle DOWN", f"{down_prob:.1%}")
    st.progress(min(1.0, max(0.0, up_prob)))
    st.caption(f"Trade threshold: {threshold:.0%}. Model: Random Forest trained on recent OHLCV-derived features.")
    st.warning("Predictions are uncertain. This small model is a learning example, not evidence of a profitable strategy.")
    st.subheader("Account snapshot")
    st.write(f"**Unrealized position value:** {fmt_money(position_value)}")
    st.write(f"**Total paper P/L:** {fmt_money(pnl)}")
    st.write(f"**Simulated trade count:** {len(st.session_state.trades)}")

st.subheader("Paper equity history")
hist = pd.DataFrame(st.session_state.equity_history)
if not hist.empty:
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=hist["time"], y=hist["equity"], mode="lines", name="Equity"))
    eq_fig.add_trace(go.Scatter(x=hist["time"], y=hist["cash"], mode="lines", name="Cash"))
    eq_fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10),
                         template="plotly_dark", yaxis_title="USD")
    st.plotly_chart(eq_fig, use_container_width=True)
else:
    st.info("Equity history will appear as the dashboard updates.")

st.subheader("Trade log")
if st.session_state.trades:
    trades_df = pd.DataFrame(st.session_state.trades)
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
else:
    st.info("No simulated trades yet. Enable 'Allow bot to simulate trades' in the sidebar. It only considers a new trade once per fetched candle.")

st.caption(
    f"Last dashboard update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • "
    "Data may be delayed or unavailable. All orders are simulated locally."
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
