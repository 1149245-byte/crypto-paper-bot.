# AI Crypto Paper Trader

A local Streamlit dashboard for **simulated crypto trading** using public exchange candles and a small Random Forest classifier. It does not place real orders and does not require exchange API keys.

## Requirements
- Python 3.10 or newer
- Internet connection for public exchange market data

## Run it
Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard should open in your browser, usually at `http://localhost:8501`.

## How to use
1. Choose an exchange, pair, and candle timeframe.
2. Start with `Allow bot to simulate trades` turned **off** to inspect signals.
3. Set starting paper cash and simulated fees.
4. Turn on simulated trading when ready. The app only evaluates execution once per newly fetched candle.
5. Use `Reset paper account` to start over.

## Important limitations
- The classifier is a teaching/demo model trained on recent candle-derived features. It is not a proven trading edge and may overfit.
- It predicts whether the next candle closes above the current close. It does not account for slippage, spreads, latency, tax, or every exchange constraint.
- The app uses Streamlit session state; the paper account is reset if the session/server restarts.
- Auto-refresh is optional. Data refresh depends on the selected exchange and network.
- Pair availability differs by exchange. If a pair is unavailable, choose a pair supported by that exchange.
- No private keys, API secrets, or real order endpoints are used.
- A green/red signal is not a recommendation to trade real money. Backtest and paper-trade for a long period before considering any live system.
