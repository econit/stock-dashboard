"""
주식 분석 대시보드
Streamlit + Plotly: 티커 선택, 기간 설정, 캔들차트 + MA + 매매신호, Raw Data, RSI 지표
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta


# 페이지 설정: 넓은 레이아웃, 제목
st.set_page_config(
    page_title="주식 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 스타일: 깔끔한 레이아웃
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .metric-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem 1.5rem; border-radius: 12px; color: white; text-align: center; }
    .stMetric label { font-size: 0.95rem !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """yfinance로 OHLCV 조회. 실패 시 None."""
    try:
        stock = yf.Ticker(ticker.strip())
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        if df is None or df.empty or len(df) < 2:
            return None
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None
        return df[required].copy()
    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """이동평균(20, 60) + RSI + 매매신호(골든/데드크로스) 추가."""
    result = df.copy()
    result["MA20"] = result["Close"].rolling(20, min_periods=1).mean()
    result["MA60"] = result["Close"].rolling(60, min_periods=1).mean()
    # RSI 14
    delta = result["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result["RSI"] = (100 - (100 / (1 + rs))).fillna(100)
    # 매매 신호: 골든크로스 = Buy, 데드크로스 = Sell
    result["Signal"] = ""
    ma20 = result["MA20"]
    ma60 = result["MA60"]
    cross_up = (ma20.shift(1) <= ma60.shift(1)) & (ma20 > ma60)
    cross_down = (ma20.shift(1) >= ma60.shift(1)) & (ma20 < ma60)
    result.loc[cross_up, "Signal"] = "Buy"
    result.loc[cross_down, "Signal"] = "Sell"
    return result


def build_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """캔들 + MA + Buy/Sell 마커 / 거래량 / RSI 를 세 개의 서브플롯으로."""
    df = df.dropna(subset=["MA20", "MA60"])
    if df.empty:
        return go.Figure()
    x = df.index
    # 3행 1열, x축 공유, 행 높이 비율: 캔들 2 : 거래량 0.8 : RSI 0.8
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("캔들 & 이동평균 / 매매신호", "거래량", "RSI (14)"),
    )
    # Row 1: 캔들 + MA + Buy/Sell
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="주가",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["MA20"], mode="lines", name="MA20", line=dict(color="#2196F3", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["MA60"], mode="lines", name="MA60", line=dict(color="#FF9800", width=2)),
        row=1,
        col=1,
    )
    buy_mask = df["Signal"] == "Buy"
    sell_mask = df["Signal"] == "Sell"
    if buy_mask.any():
        buy_df = df.loc[buy_mask]
        fig.add_trace(
            go.Scatter(
                x=buy_df.index,
                y=buy_df["Low"] * 0.998,
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(width=1, color="darkgreen")),
            ),
            row=1,
            col=1,
        )
    if sell_mask.any():
        sell_df = df.loc[sell_mask]
        fig.add_trace(
            go.Scatter(
                x=sell_df.index,
                y=sell_df["High"] * 1.002,
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=1, color="darkred")),
            ),
            row=1,
            col=1,
        )
    # Row 2: 거래량 막대
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=x, y=df["Volume"], name="거래량", marker_color=colors, showlegend=False),
        row=2,
        col=1,
    )
    # Row 3: RSI
    fig.add_trace(
        go.Scatter(x=x, y=df["RSI"], mode="lines", name="RSI", line=dict(color="#9C27B0", width=2)),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.7, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", opacity=0.7, row=3, col=1)
    # 매수 신호 날짜에 연한 초록 배경
    shapes = []
    if buy_mask.any():
        for buy_date in buy_df.index:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                x1 = buy_date + pd.Timedelta(days=1)
            else:
                x1 = buy_date
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=buy_date,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(0, 200, 83, 0.18)",
                    line=dict(width=0),
                    layer="below",
                )
            )
    # 매도 신호 날짜에 연한 빨간 배경
    if sell_mask.any():
        for sell_date in sell_df.index:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                x1 = sell_date + pd.Timedelta(days=1)
            else:
                x1 = sell_date
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=sell_date,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(239, 83, 80, 0.18)",
                    line=dict(width=0),
                    layer="below",
                )
            )
    fig.update_layout(
        title_text=f"{ticker} - 캔들 / 거래량 / RSI",
        template="plotly_white",
        height=780,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
        shapes=shapes,
    )
    fig.update_xaxes(title_text="날짜", row=3, col=1)
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    return fig


# ---------- 사이드바 ----------
st.sidebar.markdown("### ⚙️ 설정")
ticker = st.sidebar.text_input(
    "종목 티커",
    value="GOOG",
    placeholder="예: GOOG, 005380.KS, AAPL",
    help="yfinance 티커 심볼을 입력하세요.",
)
end_default = datetime.now()
start_default = end_default - timedelta(days=365)
start_date = st.sidebar.date_input("시작일", value=start_default)
end_date = st.sidebar.date_input("종료일", value=end_default)
run_analysis = st.sidebar.button("📊 분석 시작", type="primary", use_container_width=True)

# ---------- 메인 ----------
st.markdown('<p class="main-header">📈 주식 분석 대시보드</p>', unsafe_allow_html=True)
st.caption("종목 티커와 기간을 선택한 뒤 사이드바에서 **분석 시작**을 눌러주세요.")

if run_analysis:
    if not ticker or not ticker.strip():
        st.error("종목 티커를 입력해 주세요.")
    elif start_date > end_date:
        st.error("시작일은 종료일보다 이전이어야 합니다.")
    else:
        with st.spinner("데이터를 불러오는 중..."):
            df_raw = fetch_data(ticker.strip(), str(start_date), str(end_date))
        if df_raw is None:
            st.error(f"'{ticker}'에 대한 데이터를 가져오지 못했습니다. 티커와 기간을 확인해 주세요.")
        else:
            df = add_indicators(df_raw)
            current_rsi = float(df["RSI"].iloc[-1]) if len(df) else 0

            tab1, tab2 = st.tabs(["📉 차트", "📋 데이터"])

            with tab1:
                fig = build_candlestick_chart(df, ticker.strip())
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col2:
                    rsi_color = "#26a69a" if 30 <= current_rsi <= 70 else "#ef5350"
                    st.metric(
                        label="현재 RSI (14)",
                        value=f"{current_rsi:.1f}",
                        delta="과매수 주의" if current_rsi > 70 else ("과매도 관심" if current_rsi < 30 else "중립"),
                    )

            with tab2:
                display_df = df.copy()
                display_df.index.name = "Date"
                st.dataframe(display_df, use_container_width=True, height=400)

            st.success(f"**{ticker}** | {start_date} ~ {end_date} | {len(df)}일 데이터 로드 완료.")
else:
    st.info("👈 사이드바에서 **종목 티커**, **시작일/종료일**을 선택하고 **분석 시작** 버튼을 눌러주세요.")
