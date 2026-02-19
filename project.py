import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="AI Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# =======================
# FIND WORKING SYMBOL
# =======================
def find_working_symbol(company_name):
    possible_symbols = [
        company_name.replace(" ", ""),
        company_name.replace(" ", "").upper(),
        company_name.split()[0].upper(),
        company_name.split()[0].upper() + ".NS",
        company_name.replace(" ", "").upper() + ".NS"
    ]

    for sym in possible_symbols:
        try:
            df = yf.Ticker(sym).history(period="1mo")
            if not df.empty:
                return sym
        except:
            pass

    return None

# =======================
# LOAD STOCK
# =======================
@st.cache_data(ttl=600)
def load_stock(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    info = stock.info
    return df, info

# =======================
# INDICATORS
# =======================
def add_indicators(df):
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    return df.dropna()

# =======================
# ML MODEL
# =======================
class DirectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()

    def train_predict(self, df):
        df["Returns"] = df["Close"].pct_change()
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        features = ["RSI", "MACD", "MACD_signal", "SMA_20", "SMA_50", "ATR", "Momentum"]
        df = df.dropna()

        X = df[features]
        y = df["Target"]

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        acc = self.model.score(X_test, y_test)

        prob = self.model.predict_proba(
            self.scaler.transform(X.iloc[[-1]])
        )[0][1]

        return acc, prob

# =======================
# CHART
# =======================
def price_chart(df, symbol):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Price", "RSI", "MACD")
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"]), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=850)
    return fig

# =======================
# MAIN APP
# =======================
def main():
    st.title("🚀 AI Stock Dashboard (LIVE FIXED)")

    st.sidebar.header("Company Name Input")
    company = st.sidebar.text_input(
        "Enter Company Name",
        value="Reliance"
    )

    period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y"], index=1)

    symbol = find_working_symbol(company)

    if symbol is None:
        st.error("❌ Company symbol not found. Try short name (Reliance, Tata, Apple)")
        return

    st.success(f"✅ Live Symbol Found: {symbol}")

    df, info = load_stock(symbol, period)

    if df.empty:
        st.error("❌ Live data not available")
        return

    df = add_indicators(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${df.Close.iloc[-1]:.2f}")
    col2.metric("RSI", f"{df.RSI.iloc[-1]:.1f}")
    col3.metric("Market Cap", f"${info.get('marketCap', 0) / 1e9:.1f}B")

    st.plotly_chart(price_chart(df, symbol), use_container_width=True)

    model = DirectionModel()
    acc, prob = model.train_predict(df)

    st.subheader("🤖 AI Prediction")
    st.metric("Accuracy", f"{acc*100:.1f}%")
    st.metric("Tomorrow UP Probability", f"{prob*100:.1f}%")

    if prob > 0.55:
        st.success("🟢 BUY")
    elif prob > 0.50:
        st.info("🟡 HOLD")
    else:
        st.warning("🔴 WAIT")

    st.caption("⚠️ Educational purpose only")

if __name__ == "__main__":
    main()


