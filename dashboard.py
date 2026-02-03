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


@st.cache_data(ttl=3600)  # 1시간 동안 캐시 유지
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


@st.cache_data(ttl=3600)
def fetch_financial_data(ticker: str):
    """지표(info)와 재무제표(financials) 조회. 캐싱 처리."""
    try:
        stock_obj = yf.Ticker(ticker.strip())
        return stock_obj.info, stock_obj.financials
    except Exception:
        return None, None


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


def get_ytd_return(ticker: str) -> float:
    """올해 초부터 현재까지의 수익률(YTD) 계산."""
    try:
        current_year = datetime.now().year
        first_day = f"{current_year}-01-01"
        stock = yf.Ticker(ticker)
        df = stock.history(start=first_day)
        if df.empty:
            return None
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        return (end_price - start_price) / start_price * 100
    except Exception:
        return None


def build_comparison_chart(ticker1: str, ticker2: str) -> go.Figure:
    """두 회사의 6개월 주가 변동률(%) 비교 차트."""
    try:
        end = datetime.now()
        start = end - timedelta(days=180)
        
        df1 = yf.Ticker(ticker1).history(start=start, end=end)
        df2 = yf.Ticker(ticker2).history(start=start, end=end)
        
        if df1.empty or df2.empty:
            return None
            
        # 정규화 (첫날을 0%로)
        df1_norm = (df1["Close"] / df1["Close"].iloc[0] - 1) * 100
        df2_norm = (df2["Close"] / df2["Close"].iloc[0] - 1) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1_norm.index, y=df1_norm, mode="lines", name=f"{ticker1} (%)"))
        fig.add_trace(go.Scatter(x=df2_norm.index, y=df2_norm, mode="lines", name=f"{ticker2} (%)"))
        
        fig.update_layout(
            title="최근 6개월 상대 수익률 비교 (Normalized)",
            template="plotly_white",
            xaxis_title="날짜",
            yaxis_title="변동률 (%)",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception:
        return None


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

            tab1, tab2, tab3, tab4 = st.tabs(["📉 차트", "📋 데이터", "📊 재무 분석", "⚖️ 경쟁사 비교"])

            with tab1:
                # ... (보존)
                fig = build_candlestick_chart(df, ticker.strip())
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric(
                        label="현재 RSI (14)",
                        value=f"{current_rsi:.1f}",
                        delta="과매수 주의" if current_rsi > 70 else ("과매도 관심" if current_rsi < 30 else "중립"),
                    )

            with tab2:
                display_df = df.copy()
                display_df.index.name = "Date"
                st.dataframe(display_df, use_container_width=True, height=400)

            with tab3:
                st.subheader(f"🔍 {ticker} 핵심 지표")
                info, financials = fetch_financial_data(ticker.strip())
                
                if info and any(k in info for k in ["marketCap", "forwardPE", "trailingPE", "priceToBook", "returnOnEquity", "dividendYield"]):
                    try:
                        # 지표 추출
                        mkt_cap = info.get("marketCap")
                        per = info.get("forwardPE") or info.get("trailingPE")
                        pbr = info.get("priceToBook")
                        roe = info.get("returnOnEquity")
                        div_yield = info.get("dividendYield")

                        # 상단 메트릭 5개 컬럼
                        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                        
                        with m_col1:
                            if mkt_cap:
                                st.metric("시가총액", f"{mkt_cap/1e12:.2f}조")
                            else:
                                st.metric("시가총액", "N/A")
                        
                        with m_col2:
                            st.metric("PER", f"{per:.2f}" if per else "N/A")
                        
                        with m_col3:
                            st.metric("PBR", f"{pbr:.2f}" if pbr else "N/A")
                        
                        with m_col4:
                            st.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
                        
                        with m_col5:
                            st.metric("배당수익률", f"{div_yield*100:.2f}%" if div_yield else "N/A")

                        st.markdown("---")
                        st.subheader("📅 연간 실적 추이 (최근 4년)")
                        
                        if financials is not None and not financials.empty:
                            # 매출액(Total Revenue)과 순이익(Net Income) 추출
                            rev_key = "Total Revenue"
                            net_key = "Net Income"
                            
                            if rev_key in financials.index and net_key in financials.index:
                                hist_df = financials.loc[[rev_key, net_key]].T
                                hist_df.index = hist_df.index.year # 년도만 표시
                                hist_df = hist_df.sort_index().tail(4) # 최근 4년
                                
                                fig_fin = go.Figure()
                                fig_fin.add_trace(go.Bar(
                                    x=hist_df.index,
                                    y=hist_df[rev_key],
                                    name="매출액",
                                    marker_color="#636EFA"
                                ))
                                fig_fin.add_trace(go.Bar(
                                    x=hist_df.index,
                                    y=hist_df[net_key],
                                    name="순이익",
                                    marker_color="#EF553B"
                                ))
                                
                                fig_fin.update_layout(
                                    barmode='group',
                                    template="plotly_white",
                                    xaxis_title="연도",
                                    yaxis_title="금액 (USD)",
                                    height=450,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig_fin, use_container_width=True)
                            else:
                                st.warning("매출액 또는 순이익 데이터를 찾을 수 없습니다.")
                        else:
                            st.warning("연간 실적 데이터를 불러올 수 없습니다.")
                            
                    except Exception as e:
                        st.error("재무 정보를 불러오는 중 오류가 발생했습니다.")
                        st.caption(f"오류 상세: {e}")
                else:
                    st.warning("재무 정보를 불러올 수 없습니다.")
                    st.info("ETF나 일부 종목은 상세 재무 정보를 제공하지 않을 수 있습니다.")

            with tab4:
                st.subheader("⚖️ 경쟁사 비교 분석")
                
                # 경쟁사 매핑 딕셔너리 (25+ 쌍)
                peer_map = {
                    # Big Tech / Internet
                    'AAPL': 'MSFT', 'MSFT': 'AAPL',
                    'GOOG': 'META', 'META': 'GOOG',
                    'GOOGL': 'META',
                    'AMZN': 'WMT', 'WMT': 'AMZN',
                    'NFLX': 'DIS', 'DIS': 'NFLX',
                    
                    # Semiconductor
                    'NVDA': 'AMD', 'AMD': 'NVDA',
                    'TSM': 'INTC', 'INTC': 'TSM',
                    'ASML': 'AMAT', 'AMAT': 'ASML',
                    'AVGO': 'QCOM', 'QCOM': 'AVGO',
                    'MU': 'WDC', 'WDC': 'MU',
                    'LRCX': 'AMAT',
                    
                    # Automotive / EV
                    'TSLA': 'RIVN', 'RIVN': 'TSLA',
                    'TM': 'HMC', 'HMC': 'TM',
                    'F': 'GM', 'GM': 'F',
                    
                    # Finance / Payment
                    'V': 'MA', 'MA': 'V',
                    'JPM': 'BAC', 'BAC': 'JPM',
                    'GS': 'MS', 'MS': 'GS',
                    
                    # Consumer / Food
                    'KO': 'PEP', 'PEP': 'KO',
                    'NKE': 'ADDYY', 'ADDYY': 'NKE',
                    'MCD': 'SBUX', 'SBUX': 'MCD',
                    'COST': 'TGT', 'TGT': 'COST',
                    
                    # Korea Market (KOSPI/KOSDAQ)
                    '005930.KS': '000660.KS', '000660.KS': '005930.KS', # 삼성전자 - SK하이닉스
                    '005380.KS': '000270.KS', '000270.KS': '005380.KS', # 현대차 - 기아
                    '035420.KS': '035720.KS', '035720.KS': '035420.KS', # NAVER - 카카오
                    '068270.KS': '207940.KS', '207940.KS': '068270.KS', # 셀트리온 - 삼성바이오로직스
                    '373220.KS': '006400.KS', '006400.KS': '373220.KS', # LG에너지솔루션 - 삼성SDI
                    '005490.KS': '010130.KS', '010130.KS': '005490.KS', # POSCO홀딩스 - 고려아연
                    '051910.KS': '010950.KS', '010950.KS': '051910.KS', # LG화학 - S-Oil
                    '000270.KS': '005380.KS' # 기아 - 현대차 (중복 방지용 확인)
                }
                
                base_ticker = ticker.strip().upper()
                suggested_peer = peer_map.get(base_ticker, "")
                
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    peer_ticker = st.text_input("비교할 경쟁사 티커를 입력하세요:", value=suggested_peer, key="peer_input").strip().upper()
                
                if peer_ticker:
                    with st.spinner(f"{base_ticker} vs {peer_ticker} 비교 중..."):
                        info1, _ = fetch_financial_data(base_ticker)
                        info2, _ = fetch_financial_data(peer_ticker)
                        
                        if info1 and info2:
                            # 데이터 추출
                            def extract_metrics(info, t):
                                return {
                                    'Ticker': t,
                                    'PER': info.get("forwardPE") or info.get("trailingPE"),
                                    'PBR': info.get("priceToBook"),
                                    'ROE': (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None,
                                    'YTD': get_ytd_return(t)
                                }
                            
                            m1 = extract_metrics(info1, base_ticker)
                            m2 = extract_metrics(info2, peer_ticker)
                            
                            # 비교표 시각화
                            comp_data = {
                                "지표": ["PER (낮을수록 우수)", "PBR (낮을수록 우수)", "ROE (%) (높을수록 우수)", "YTD 수익률 (%) (높을수록 우수)"],
                                base_ticker: [m1['PER'], m1['PBR'], m1['ROE'], m1['YTD']],
                                peer_ticker: [m2['PER'], m2['PBR'], m2['ROE'], m2['YTD']]
                            }
                            comp_df = pd.DataFrame(comp_data)
                            
                            # 하이라이트 함수 (PER, PBR은 낮은 것, ROE, YTD는 높은 것)
                            def highlight_better(s):
                                if s.name == "지표": return [''] * len(s)
                                res = []
                                for i, val in enumerate(s):
                                    other_val = comp_df.iloc[i, 2 if s.name == base_ticker else 1]
                                    if val is None or other_val is None:
                                        res.append('')
                                        continue
                                    
                                    is_better = False
                                    if i < 2: # PER, PBR (낮을수록 좋음)
                                        if val < other_val: is_better = True
                                    else: # ROE, YTD (높을수록 좋음)
                                        if val > other_val: is_better = True
                                        
                                    res.append('background-color: rgba(38, 166, 154, 0.3)' if is_better else '')
                                return res

                            st.table(comp_df.style.apply(highlight_better).format({base_ticker: "{:.2f}", peer_ticker: "{:.2f}"}))
                            
                            # 비교 차트
                            fig_comp = build_comparison_chart(base_ticker, peer_ticker)
                            if fig_comp:
                                st.plotly_chart(fig_comp, use_container_width=True)
                            else:
                                st.error("수익률 비교 차트를 생성할 수 없습니다.")
                        else:
                            st.error("경쟁사 정보를 불러올 수 없습니다. 티커를 확인해 주세요.")
                else:
                    st.info("비교할 경쟁사 티커를 입력해 주세요.")

            st.success(f"**{ticker}** | {start_date} ~ {end_date} | {len(df)}일 데이터 로드 완료.")
else:
    st.info("👈 사이드바에서 **종목 티커**, **시작일/종료일**을 선택하고 **분석 시작** 버튼을 눌러주세요.")
