import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="도로 소통정보 분석실",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- 디자인 --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&family=Jua&display=swap');
html, body, [class*="css"] {font-family:'Gowun Dodum', sans-serif;}
h1,h2,h3 {font-family:'Jua', sans-serif;}
.stApp {background: linear-gradient(180deg,#f4f8ff 0%,#f8fbff 42%,#f7faf9 100%);}
.block-container {padding-top:1.5rem; padding-bottom:2rem; max-width:1500px;}
.hero {background:linear-gradient(120deg,#152e55,#246b82);padding:26px 30px;border-radius:22px;color:white;margin-bottom:18px;box-shadow:0 10px 28px rgba(24,56,90,.14);}
.hero h1 {color:white;font-size:2.15rem;margin:0 0 8px 0;}
.hero p {color:#e4f3ff;font-size:1rem;margin:0;}
.mini {background:white;border:1px solid #e2eaf3;border-radius:18px;padding:18px 20px;box-shadow:0 4px 14px rgba(30,60,90,.04);}
.metric-label {color:#60758b;font-size:.9rem;margin-bottom:6px;}
.metric-value {font-size:1.7rem;font-weight:800;color:#173c5a;}
.metric-note {color:#8193a5;font-size:.8rem;margin-top:5px;}
.section-note {background:#eaf4f7;border-left:4px solid #31849a;padding:11px 14px;border-radius:8px;color:#264e61;margin:8px 0 16px 0;}
.pill {display:inline-block;background:#e2f2ec;color:#26705c;padding:4px 10px;border-radius:99px;font-size:.8rem;margin-right:5px;}
div[data-testid="stMetric"] {background:white;border:1px solid #e2eaf3;padding:15px 18px;border-radius:16px;box-shadow:0 4px 14px rgba(30,60,90,.04);}
div[data-testid="stTabs"] button {font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚦 도로 소통정보 분석실</h1>
  <p>1시간 단위 도로 속도 데이터를 직접 살펴보고, 시간대별 차이부터 회귀·군집 분석까지 해보는 데이터 탐험 공간</p>
</div>
""", unsafe_allow_html=True)

# -------------------- 데이터 불러오기 --------------------
def read_csv_safely(file):
    raw = file.getvalue()
    errors = []
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        for sep in [None, ",", "\t", ";"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                errors.append(str(e))
    raise ValueError("CSV를 읽지 못했어요. 파일 인코딩이나 구분자를 확인해 주세요.")

def norm(s):
    return re.sub(r"[\s_\-()/\[\]]", "", str(s)).lower()

def guess_col(cols, keywords):
    scored = []
    for c in cols:
        nc = norm(c)
        score = sum(1 for k in keywords if k in nc)
        if score:
            scored.append((score, c))
    return sorted(scored, reverse=True, key=lambda x: x[0])[0][1] if scored else None

def parse_hour(value):
    s = str(value).strip()
    # '~1시', '01시', '1:00', '2026-10-20 01:00:00' 등의 형식 지원
    m = re.search(r"(?:~\s*)?(\d{1,2})\s*시", s)
    if m:
        h = int(m.group(1))
        return 0 if h == 24 else h
    m = re.search(r"(\d{1,2})\s*:\s*\d{2}", s)
    if m:
        return int(m.group(1))
    try:
        dt = pd.to_datetime(s, errors="raise")
        return int(dt.hour)
    except Exception:
        return np.nan

def numericize(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("km/h", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")

uploaded = st.sidebar.file_uploader("📂 1시간 소통정보 CSV 올리기", type=["csv", "txt"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 분석 안내")
st.sidebar.caption("시간은 CSV의 B열(두 번째 열)에 있는 1~24 시간코드를 기준으로 분석해요. 나머지 컬럼은 자동으로 인식해요.")
st.sidebar.caption("추천: 도로명, 날짜, 시간대, 평균속도(또는 소통속도)가 들어 있는 CSV")

if uploaded is None:
    st.info("👋 먼저 **`1시간 소통정보 (도로별-방향별).csv`** 파일을 왼쪽에서 올려 주세요. 파일을 올리면 분석 화면이 열려요.")
    st.markdown("### 이 앱에서 볼 수 있는 것")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown('<div class="mini"><h3>📊 기초통계</h3><p>평균·중앙값·최솟값·최댓값·표준편차·사분위수</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="mini"><h3>🕒 시간대 분석</h3><p>새벽·출근·낮·퇴근·밤의 속도 차이</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="mini"><h3>📈 회귀 분석</h3><p>시간과 속도의 관계, 추세선, R² 확인</p></div>', unsafe_allow_html=True)
    c4.markdown('<div class="mini"><h3>🧩 군집 분석</h3><p>속도 패턴이 비슷한 도로를 자동으로 묶기</p></div>', unsafe_allow_html=True)
    st.stop()

try:
    raw_df = read_csv_safely(uploaded)
except Exception as e:
    st.error(f"파일을 읽는 중 문제가 생겼어요: {e}")
    st.stop()

if raw_df.empty:
    st.error("읽어온 데이터가 비어 있어요. CSV 파일을 확인해 주세요.")
    st.stop()

# -------------------- 컬럼 자동 추정 + 사용자 선택 --------------------
cols = list(raw_df.columns)
time_guess = guess_col(cols, ["시간", "시각", "hour", "time", "기준일시", "일시"])
date_guess = guess_col(cols, ["날짜", "일자", "date", "기준일"])
road_guess = guess_col(cols, ["도로명", "도로", "노선명", "road", "구간명"])
speed_guess = guess_col(cols, ["평균속도", "속도", "speed", "소통속도", "통행속도"])
type_guess = guess_col(cols, ["도로유형", "도로종류", "도로구분", "유형", "type"])
direction_guess = guess_col(cols, ["방향", "상하행", "진행방향"])

st.sidebar.markdown("### 🔧 컬럼 확인")
def choose(label, guess, allow_none=True):
    opts = (["선택 안 함"] if allow_none else []) + cols
    default = guess if guess in cols else ("선택 안 함" if allow_none else cols[0])
    idx = opts.index(default) if default in opts else 0
    val = st.sidebar.selectbox(label, opts, index=idx)
    return None if val == "선택 안 함" else val

road_col = choose("도로명 컬럼", road_guess)
date_col = choose("날짜 컬럼", date_guess)
# 이 데이터 파일은 B열(두 번째 열)에 1~24의 시간코드가 들어 있음
if len(raw_df.columns) >= 2:
    time_col = raw_df.columns[1]
    st.sidebar.caption(f"시간 코드: B열 · {time_col} (1~24)")
else:
    st.error("CSV 파일에서 B열 시간코드를 찾을 수 없습니다.")
    st.stop()
speed_col = choose("속도 컬럼", speed_guess, allow_none=False)
type_col = choose("도로 유형 컬럼", type_guess)
direction_col = choose("방향 컬럼", direction_guess)

df = raw_df.copy()
df["_속도"] = numericize(df[speed_col])
df["_속도"] = df["_속도"].replace([np.inf, -np.inf], np.nan)
df = df[df["_속도"].notna()].copy()

if df.empty:
    st.error(f"'{speed_col}' 컬럼에서 숫자 속도를 찾지 못했어요. 속도 컬럼을 다시 선택해 주세요.")
    st.stop()

if date_col:
    # YYYYMMDD 숫자형 날짜와 일반 날짜 문자열을 모두 처리
    date_values = df[date_col]
    if pd.api.types.is_numeric_dtype(date_values):
        date_text = date_values.astype("Int64").astype(str)
        df["_날짜"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    else:
        date_text = date_values.astype(str).str.strip()
        compact_date = date_text.str.fullmatch(r"\d{8}")
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        parsed.loc[compact_date] = pd.to_datetime(date_text.loc[compact_date], format="%Y%m%d", errors="coerce")
        parsed.loc[~compact_date] = pd.to_datetime(date_text.loc[~compact_date], errors="coerce")
        df["_날짜"] = parsed
else:
    df["_날짜"] = pd.NaT

# B열 시간코드 1~24를 실제 시간 0~23시로 변환
# 예: 코드 1 = 00~01시, 코드 24 = 23~24시
df["_시간코드"] = pd.to_numeric(df[time_col], errors="coerce")
valid_time_code = df["_시간코드"].between(1, 24, inclusive="both")
df["_시"] = np.where(valid_time_code, df["_시간코드"] - 1, np.nan)
df["_시"] = pd.to_numeric(df["_시"], errors="coerce")

# 시간 구간은 시간 값에 따라 자동 분류
def time_group(h):
    if pd.isna(h): return "시간 정보 없음"
    h = int(h) % 24
    if 0 <= h <= 5: return "🌙 새벽 (00–05)"
    if 6 <= h <= 9: return "🚗 출근 시간 (06–09)"
    if 10 <= h <= 16: return "☀️ 낮 시간 (10–16)"
    if 17 <= h <= 19: return "🚦 퇴근 시간 (17–19)"
    return "🌆 밤 시간 (20–23)"

df["_시간대"] = df["_시"].map(time_group)
df["_도로명"] = df[road_col].astype(str) if road_col else "도로명 정보 없음"
df["_도로유형"] = df[type_col].astype(str) if type_col else "도로 유형 정보 없음"
df["_방향"] = df[direction_col].astype(str) if direction_col else "방향 정보 없음"

# 시간 값이 ~1시처럼 끝나는 시간 표시라면 24시는 0시로 처리됨. 앱 내 안내를 표시
st.markdown(f'<span class="pill">시간 코드: B열 · {time_col}</span><span class="pill">속도 컬럼: {speed_col}</span><span class="pill">유효 시간 코드: {int(valid_time_code.sum()):,}개</span><span class="pill">유효 속도 데이터: {len(df):,}개</span>', unsafe_allow_html=True)

# -------------------- 필터 --------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 데이터 골라보기")
roads = sorted(df["_도로명"].dropna().unique().tolist())
if len(roads) > 1 and road_col:
    selected_roads = st.sidebar.multiselect("도로 선택 (비우면 전체)", roads, default=[])
else:
    selected_roads = []
types = sorted(df["_도로유형"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect("도로 유형 선택 (비우면 전체)", types, default=[])
groups = ["🌙 새벽 (00–05)", "🚗 출근 시간 (06–09)", "☀️ 낮 시간 (10–16)", "🚦 퇴근 시간 (17–19)", "🌆 밤 시간 (20–23)", "시간 정보 없음"]
selected_groups = st.sidebar.multiselect("시간대 선택 (비우면 전체)", groups, default=[])

view = df.copy()
if selected_roads:
    view = view[view["_도로명"].isin(selected_roads)]
if selected_types:
    view = view[view["_도로유형"].isin(selected_types)]
if selected_groups:
    view = view[view["_시간대"].isin(selected_groups)]
if view.empty:
    st.warning("선택한 조건에 맞는 데이터가 없어요. 필터를 조금 풀어 주세요.")
    st.stop()

# -------------------- 상단 지표 --------------------
st.markdown("## 👀 먼저 데이터 한눈에 보기")
n = len(view)
mean_v = view["_속도"].mean()
median_v = view["_속도"].median()
std_v = view["_속도"].std()
missing = int(raw_df.isna().sum().sum())
dupes = int(raw_df.duplicated().sum())
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("🚘 분석한 기록", f"{n:,}개", f"원본 {len(raw_df):,}개 행")
c2.metric("평균 속도", f"{mean_v:,.2f}", "속도 단위는 원본 데이터 기준")
c3.metric("중앙값", f"{median_v:,.2f}", "절반의 값이 이보다 작거나 같아요")
c4.metric("표준편차", f"{std_v:,.2f}" if pd.notna(std_v) else "계산 불가", "값이 클수록 속도 차이가 큼")
c5.metric("누락 셀", f"{missing:,}", "원본 전체 기준")

st.markdown('<div class="section-note">💡 <b>읽는 팁:</b> 평균은 전체적인 수준을, 중앙값은 가운데 값을 보여줘요. 평균과 중앙값 차이가 크면 일부 매우 빠르거나 느린 기록의 영향을 받았을 수 있어요. 아래 그래프를 함께 보면서 확인해 보세요.</div>', unsafe_allow_html=True)

tabs = st.tabs(["📌 요약 & 기초통계", "🕒 시간대 패턴", "🛣️ 도로별 비교", "📈 회귀 분석", "🧩 군집 분석", "🧾 원본 데이터"])

# -------------------- 탭 1 --------------------
with tabs[0]:
    left,right = st.columns([1.05,1.4], gap="large")
    with left:
        st.subheader("기초통계 요약")
        stats = view["_속도"].describe(percentiles=[.05,.25,.5,.75,.95]).rename({
            "count":"자료 수", "mean":"평균", "std":"표준편차", "min":"최솟값",
            "5%":"5% 백분위", "25%":"1사분위수(Q1)", "50%":"중앙값(Q2)",
            "75%":"3사분위수(Q3)", "95%":"95% 백분위", "max":"최댓값"
        })
        stats_df = stats.to_frame("속도").round(3)
        st.dataframe(stats_df, use_container_width=True)
        st.download_button("⬇️ 기초통계 CSV 저장", stats_df.to_csv(encoding="utf-8-sig").encode("utf-8-sig"), "기초통계.csv", "text/csv")
        q1, q3 = view["_속도"].quantile([.25,.75])
        st.write(f"**중간 50%의 범위(IQR)**: {q1:.2f} ~ {q3:.2f} (폭 {q3-q1:.2f})")
        if mean_v > median_v:
            st.caption("평균이 중앙값보다 높아요. 일부 높은 속도 기록이 평균을 끌어올렸을 가능성이 있어요.")
        elif mean_v < median_v:
            st.caption("평균이 중앙값보다 낮아요. 일부 낮은 속도 기록이 평균을 끌어내렸을 가능성이 있어요.")
        else:
            st.caption("평균과 중앙값이 비슷해요. 다만 이것만으로 분포가 완전히 대칭이라고 단정할 수는 없어요.")
    with right:
        st.subheader("속도 분포 살펴보기")
        fig = px.histogram(view, x="_속도", nbins=35, marginal="box",
                           labels={"_속도":"속도", "count":"기록 수"},
                           title="속도는 어느 구간에 가장 많이 모여 있을까?",
                           color_discrete_sequence=["#2c8c99"])
        fig.update_layout(template="plotly_white", height=410, bargap=.06, xaxis_title="속도", yaxis_title="기록 수")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Box(x=view["_속도"], name="속도 분포", boxmean="sd", marker_color="#e78b52"))
        fig2.update_layout(template="plotly_white", title="상자수염그림으로 보는 퍼짐과 이상치", xaxis_title="속도", height=250)
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("숫자형 컬럼 상관관계")
    numeric_cols = [c for c in view.columns if pd.api.types.is_numeric_dtype(view[c]) and c != "_속도"]
    corr_options = ["_속도"] + numeric_cols
    corr_options = list(dict.fromkeys(corr_options))
    if len(corr_options) >= 2:
        corr = view[corr_options].corr(numeric_only=True)
        figc = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="상관계수 히트맵 (-1에 가까움: 음의 관계 / +1에 가까움: 양의 관계)")
        figc.update_layout(height=450)
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("상관관계를 계산할 다른 숫자형 컬럼이 없어요. 시간대 분석과 회귀 분석을 이용해 보세요.")

# -------------------- 탭 2 --------------------
with tabs[1]:
    st.subheader("시간대별 평균 속도")
    if view["_시"].notna().any():
        hourly = view.dropna(subset=["_시"]).groupby("_시")["_속도"].agg(["mean","median","count","std"]).reset_index()
        hourly["_시"] = hourly["_시"].astype(int)
        hourly["시간"] = hourly["_시"].map(lambda h: f"{h:02d}시")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly["시간"], y=hourly["mean"], mode="lines+markers", name="평균 속도",
                                 line=dict(color="#247f91", width=3), marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=hourly["시간"], y=hourly["median"], mode="lines+markers", name="중앙값",
                                 line=dict(color="#e78b52", width=2, dash="dot")))
        fig.update_layout(template="plotly_white", title="시간이 바뀌면 속도도 달라질까?", xaxis_title="시간", yaxis_title="속도", height=400)
        st.plotly_chart(fig, use_container_width=True)
        group_order = ["🌙 새벽 (00–05)", "🚗 출근 시간 (06–09)", "☀️ 낮 시간 (10–16)", "🚦 퇴근 시간 (17–19)", "🌆 밤 시간 (20–23)", "시간 정보 없음"]
        by_group = view.groupby("_시간대")["_속도"].agg(평균="mean", 중앙값="median", 기록수="count", 표준편차="std").reindex(group_order).dropna(how="all").reset_index()
        figg = px.bar(by_group, x="_시간대", y="평균", color="평균", color_continuous_scale="Teal",
                      text=by_group["평균"].map(lambda x: f"{x:.1f}"), title="시간대별 평균 속도 비교",
                      hover_data={"중앙값":":.2f","기록수":True,"표준편차":":.2f"})
        figg.update_layout(template="plotly_white", xaxis_title="시간대", yaxis_title="평균 속도", coloraxis_showscale=False, height=410)
        st.plotly_chart(figg, use_container_width=True)
        st.dataframe(by_group.round(2), use_container_width=True, hide_index=True)
        st.caption("※ 이 CSV는 B열 시간코드 1~24를 사용해요. 코드 1은 00~01시(그래프 00시), 코드 2는 01~02시(그래프 01시), 코드 24는 23~24시(그래프 23시)로 변환했어요.")
    else:
        st.warning("B열 시간코드에서 1~24 범위의 값을 찾지 못했어요. CSV의 두 번째 열이 시간코드인지 확인해 주세요.")

# -------------------- 탭 3 --------------------
with tabs[2]:
    st.subheader("도로별 속도 비교")
    if road_col:
        road_summary = view.groupby("_도로명")["_속도"].agg(평균="mean", 중앙값="median", 기록수="count", 표준편차="std").reset_index()
        min_n = st.slider("도로별 비교에 포함할 최소 기록 수", 1, max(2, int(road_summary["기록수"].max())), min(10, int(road_summary["기록수"].max())))
        road_summary = road_summary[road_summary["기록수"] >= min_n].sort_values("평균", ascending=False)
        if len(road_summary):
            top_n = st.slider("표시할 도로 수", 5, min(50, len(road_summary)) if len(road_summary) >= 5 else len(road_summary), min(20, len(road_summary)))
            low = road_summary.nsmallest(top_n, "평균").sort_values("평균")
            high = road_summary.nlargest(top_n, "평균").sort_values("평균", ascending=False)
            a,b = st.columns(2)
            with a:
                fig = px.bar(high, x="평균", y="_도로명", orientation="h", text=high["평균"].map(lambda x:f"{x:.1f}"),
                             title=f"평균 속도가 높은 도로 TOP {top_n}", color="평균", color_continuous_scale="Teal")
                fig.update_layout(template="plotly_white", height=max(400, top_n*24), yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with b:
                fig = px.bar(low, x="평균", y="_도로명", orientation="h", text=low["평균"].map(lambda x:f"{x:.1f}"),
                             title=f"평균 속도가 낮은 도로 TOP {top_n}", color="평균", color_continuous_scale="Oranges")
                fig.update_layout(template="plotly_white", height=max(400, top_n*24), yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(road_summary.sort_values("평균", ascending=False).round(2), use_container_width=True, hide_index=True)
            st.download_button("⬇️ 도로별 통계 저장", road_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "도로별_속도통계.csv", "text/csv")
        else:
            st.info("최소 기록 수 조건에 맞는 도로가 없어요.")
    else:
        st.info("도로명 컬럼을 선택하면 도로별 비교가 가능해요.")
    if type_col:
        st.subheader("도로 유형별 평균 속도")
        type_summary = view.groupby("_도로유형")["_속도"].agg(평균="mean", 중앙값="median", 기록수="count").reset_index().sort_values("평균", ascending=False)
        figt = px.bar(type_summary, x="_도로유형", y="평균", color="평균", text=type_summary["평균"].map(lambda x:f"{x:.1f}"),
                      title="도로 유형별 평균 속도", color_continuous_scale="Sunset")
        figt.update_layout(template="plotly_white", xaxis_title="도로 유형", yaxis_title="평균 속도", coloraxis_showscale=False, height=370)
        st.plotly_chart(figt, use_container_width=True)
        st.dataframe(type_summary.round(2), use_container_width=True, hide_index=True)

# -------------------- 탭 4 회귀 및 분류 모델 --------------------
with tabs[3]:
    st.subheader("📈 다양한 회귀·분류 모델 비교")
    st.markdown('<div class="section-note">평균속도를 예측하는 회귀 모델과 혼잡 여부를 분류하는 로지스틱 회귀를 한 화면에서 비교해요. 평가는 날짜 기준으로 학습용과 테스트용을 나눠 진행합니다.</div>', unsafe_allow_html=True)

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                                     accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix)
        from sklearn.base import clone

        model_df = view.dropna(subset=["_시", "_속도"]).copy()
        model_df["_시"] = pd.to_numeric(model_df["_시"], errors="coerce")
        model_df = model_df.dropna(subset=["_시"])
        model_df["_날짜문자"] = model_df["_날짜"].dt.strftime("%Y-%m-%d")
        model_df["_요일"] = model_df["_날짜"].dt.dayofweek.astype("Int64").astype(str)
        model_df["_시간대문자"] = model_df["_시"].astype(int).astype(str)

        if len(model_df) < 30:
            st.warning("모델 분석에는 유효한 데이터가 최소 30개 이상 필요해요.")
        else:
            # 마지막 날짜를 테스트 데이터로 사용해 이후 날짜 예측 상황을 모의
            available_dates = sorted(model_df["_날짜문자"].dropna().unique().tolist())
            if len(available_dates) >= 2:
                test_date = available_dates[-1]
                train_df = model_df[model_df["_날짜문자"] < test_date].copy()
                test_df = model_df[model_df["_날짜문자"] == test_date].copy()
                split_label = f"학습: {', '.join(available_dates[:-1])} / 테스트: {test_date}"
            else:
                # 날짜가 없거나 하루뿐이면 시간순 80:20 분할
                model_df = model_df.sort_values("_시")
                cut = int(len(model_df) * 0.8)
                train_df, test_df = model_df.iloc[:cut].copy(), model_df.iloc[cut:].copy()
                split_label = "날짜 정보가 부족해 시간순 80:20 분할 사용"

            st.caption(f"데이터 분할 방식: {split_label} · 학습 {len(train_df):,}개 / 테스트 {len(test_df):,}개")
            if len(train_df) < 10 or len(test_df) < 5:
                st.warning("학습 또는 테스트 데이터가 부족해요. 필터 조건을 넓혀 주세요.")
            else:
                # 숫자·범주형 설명변수 구성
                numeric_features = ["_시"]
                categorical_features = ["_도로유형", "_방향"]
                if road_col:
                    categorical_features.append("_도로명")
                feature_cols = numeric_features + categorical_features
                X_train = train_df[feature_cols].copy()
                X_test = test_df[feature_cols].copy()
                y_train = train_df["_속도"].astype(float)
                y_test = test_df["_속도"].astype(float)

                def make_preprocessor(scale_numeric=True):
                    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
                    if scale_numeric:
                        numeric_steps.append(("scaler", StandardScaler()))
                    numeric_pipe = Pipeline(numeric_steps)
                    categorical_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])
                    return ColumnTransformer([
                        ("num", numeric_pipe, numeric_features),
                        ("cat", categorical_pipe, categorical_features)
                    ])

                # 다항회귀는 시간 변수에만 2차항을 적용하고 범주형 변수는 원-핫 인코딩
                def make_poly_preprocessor():
                    num_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("scaler", StandardScaler())
                    ])
                    cat_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])
                    return ColumnTransformer([
                        ("num", num_pipe, numeric_features),
                        ("cat", cat_pipe, categorical_features)
                    ])

                regression_models = {
                    "선형회귀": Pipeline([("prep", make_preprocessor()), ("model", LinearRegression())]),
                    "2차 다항회귀": Pipeline([("prep", make_poly_preprocessor()), ("model", LinearRegression())]),
                    "릿지 회귀": Pipeline([("prep", make_preprocessor()), ("model", Ridge(alpha=1.0))]),
                    "라쏘 회귀": Pipeline([("prep", make_preprocessor()), ("model", Lasso(alpha=0.05, max_iter=5000))]),
                    "랜덤 포레스트 회귀": Pipeline([("prep", make_preprocessor(scale_numeric=False)),
                                             ("model", RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                                                                              random_state=42, n_jobs=-1))])
                }

                st.markdown("### ① 평균속도 예측: 회귀 모델 비교")
                st.caption("종속변수는 평균속도이며, 시간·도로 유형·방향·도로명을 이용해 테스트 날짜의 속도를 예측합니다.")
                regression_rows = []
                regression_predictions = {}
                for name, pipe in regression_models.items():
                    try:
                        pipe.fit(X_train, y_train)
                        pred = pipe.predict(X_test)
                        regression_predictions[name] = pred
                        regression_rows.append({
                            "모델": name,
                            "R²": r2_score(y_test, pred),
                            "MAE": mean_absolute_error(y_test, pred),
                            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred)))
                        })
                    except Exception as model_error:
                        st.warning(f"{name} 실행 중 오류: {model_error}")

                if regression_rows:
                    reg_results = pd.DataFrame(regression_rows).sort_values("RMSE")
                    best_reg_name = str(reg_results.iloc[0]["모델"])
                    best_reg = reg_results.iloc[0]
                    a,b,c = st.columns(3)
                    a.metric("RMSE가 가장 낮은 모델", best_reg_name)
                    b.metric("최고 모델 MAE", f"{best_reg['MAE']:.2f}")
                    c.metric("최고 모델 R²", f"{best_reg['R²']:.3f}")
                    st.dataframe(reg_results.round(4), use_container_width=True, hide_index=True)
                    metric_long = reg_results.melt(id_vars="모델", value_vars=["MAE", "RMSE"],
                                                   var_name="평가지표", value_name="오차")
                    err_fig = px.bar(metric_long, x="모델", y="오차", color="평가지표", barmode="group",
                                     title="회귀 모델별 예측 오차 비교", barmode_groupgap=0.12)
                    err_fig.update_layout(template="plotly_white", height=390, xaxis_title="모델", yaxis_title="오차 (km/h)")
                    st.plotly_chart(err_fig, use_container_width=True)

                    selected_reg = st.selectbox("실제값과 예측값을 비교할 모델", list(regression_predictions.keys()),
                                                index=list(regression_predictions.keys()).index(best_reg_name), key="reg_compare_model")
                    pred_df = pd.DataFrame({"실제 평균속도": y_test.to_numpy(),
                                            "예측 평균속도": regression_predictions[selected_reg]})
                    pred_fig = px.scatter(pred_df, x="실제 평균속도", y="예측 평균속도",
                                          title=f"{selected_reg}: 실제값과 예측값 비교",
                                          opacity=0.35)
                    low = float(min(pred_df.min().min(), 0))
                    high = float(pred_df.max().max())
                    pred_fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines",
                                                  name="완벽한 예측 기준선", line=dict(dash="dash", color="#e78b52")))
                    pred_fig.update_layout(template="plotly_white", height=430)
                    st.plotly_chart(pred_fig, use_container_width=True)
                    st.download_button("⬇️ 회귀 모델 성능 비교 CSV 저장",
                                       reg_results.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                       "회귀모델_성능비교.csv", "text/csv")
                    st.caption("R²는 높을수록, MAE와 RMSE는 낮을수록 좋아요. 서로 다른 날짜를 테스트했으므로 점수는 해당 데이터 기간에 대한 결과이며 장기 예측 성능을 보장하지 않아요.")

                st.markdown("---")
                st.markdown("### ② 혼잡 여부 분류: 로지스틱 회귀")
                st.caption("평균속도 자체가 아니라 설정한 기준보다 느린 도로를 혼잡으로 분류합니다.")
                threshold = st.slider("혼잡 판정 기준 속도 (km/h 미만이면 혼잡)", 5, 40, 20, 1, key="logistic_threshold")
                y_train_cls = (y_train < threshold).astype(int)
                y_test_cls = (y_test < threshold).astype(int)
                if y_train_cls.nunique() < 2 or y_test_cls.nunique() < 2:
                    st.warning("현재 분할에서는 혼잡 또는 비혼잡 중 한 종류만 있어 로지스틱 회귀 성능을 계산할 수 없어요.")
                else:
                    logistic_pipe = Pipeline([
                        ("prep", make_preprocessor()),
                        ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))
                    ])
                    logistic_pipe.fit(X_train, y_train_cls)
                    cls_pred = logistic_pipe.predict(X_test)
                    cls_prob = logistic_pipe.predict_proba(X_test)[:, 1]
                    cls_acc = accuracy_score(y_test_cls, cls_pred)
                    cls_precision = precision_score(y_test_cls, cls_pred, zero_division=0)
                    cls_recall = recall_score(y_test_cls, cls_pred, zero_division=0)
                    cls_f1 = f1_score(y_test_cls, cls_pred, zero_division=0)
                    try:
                        cls_auc = roc_auc_score(y_test_cls, cls_prob)
                    except ValueError:
                        cls_auc = np.nan
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("정확도", f"{cls_acc:.3f}")
                    c2.metric("정밀도", f"{cls_precision:.3f}")
                    c3.metric("재현율", f"{cls_recall:.3f}")
                    c4.metric("F1 점수", f"{cls_f1:.3f}")
                    c5.metric("AUC", f"{cls_auc:.3f}" if pd.notna(cls_auc) else "계산 불가")
                    cm = confusion_matrix(y_test_cls, cls_pred, labels=[0,1])
                    cm_df = pd.DataFrame(cm, index=["실제 비혼잡", "실제 혼잡"], columns=["예측 비혼잡", "예측 혼잡"])
                    st.markdown("#### 혼동행렬")
                    st.dataframe(cm_df, use_container_width=True)
                    st.caption("정밀도는 혼잡이라고 예측한 기록 중 실제 혼잡 비율, 재현율은 실제 혼잡 기록 중 모델이 찾아낸 비율이에요.")
                    cls_summary = pd.DataFrame([{
                        "혼잡 기준(km/h)": threshold, "정확도": cls_acc, "정밀도": cls_precision,
                        "재현율": cls_recall, "F1": cls_f1, "AUC": cls_auc
                    }])
                    st.download_button("⬇️ 로지스틱 회귀 결과 CSV 저장",
                                       cls_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                       "로지스틱회귀_성능.csv", "text/csv")

                st.markdown("---")
                st.markdown("### ③ 분석할 때 주의할 점")
                st.info("혼잡 기준 속도는 분석자가 정한 값이며 공식 혼잡 판정 기준과 다를 수 있어요. 도로명은 기존 도로의 특성을 학습하는 데 도움이 되지만, 학습에 없던 새로운 도로에 대한 예측에는 그대로 적용하기 어려울 수 있습니다. 또한 현재 데이터는 3일 자료이므로 결과를 장기적인 서울 교통 패턴으로 일반화하지 마세요.")

    except ImportError as e:
        st.error("회귀·분류 분석에는 scikit-learn이 필요해요. requirements.txt에 scikit-learn을 추가한 뒤 다시 배포해 주세요.")

# -------------------- 탭 5 군집 --------------------
with tabs[4]:
    st.subheader("🧩 군집 분석: 비슷한 도로끼리 묶어 보기")
    st.markdown('<div class="section-note">군집 분석은 정답 라벨 없이 특징이 비슷한 대상을 자동으로 묶어요. 여기서는 도로별 평균 속도·중앙값·속도 변동을 이용해 비슷한 속도 특성을 가진 도로 그룹을 찾아요.</div>', unsafe_allow_html=True)
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        if not road_col:
            st.warning("도로명 컬럼을 선택해야 도로별 군집을 만들 수 있어요.")
        else:
            cluster_base = view.groupby("_도로명").agg(
                평균속도=("_속도","mean"),
                중앙속도=("_속도","median"),
                속도표준편차=("_속도","std"),
                기록수=("_속도","count")
            ).reset_index().fillna(0)
            cluster_base = cluster_base[cluster_base["기록수"] >= 2].copy()
            if len(cluster_base) < 3:
                st.warning("군집 분석에는 기록이 있는 도로가 최소 3개 이상 필요해요.")
            else:
                k = st.slider("군집 수 (K)", 2, min(8, len(cluster_base)-1), min(3, len(cluster_base)-1), key="cluster_k")
                features = ["평균속도","중앙속도","속도표준편차"]
                scaler = StandardScaler()
                X = scaler.fit_transform(cluster_base[features])
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_base["군집"] = model.fit_predict(X) + 1
                pca = PCA(n_components=2, random_state=42)
                pcs = pca.fit_transform(X)
                cluster_base["PCA 1"] = pcs[:,0]
                cluster_base["PCA 2"] = pcs[:,1]
                fig = px.scatter(cluster_base, x="PCA 1", y="PCA 2", color=cluster_base["군집"].astype(str),
                                 hover_name="_도로명", hover_data=features+["기록수"],
                                 title="도로별 속도 특성 군집 지도(2차원으로 압축해 표시)",
                                 labels={"color":"군집"}, color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(template="plotly_white", height=500, legend_title="군집")
                st.plotly_chart(fig, use_container_width=True)
                cluster_summary = cluster_base.groupby("군집").agg(
                    도로수=("_도로명","count"), 평균속도=("평균속도","mean"),
                    중앙속도=("중앙속도","mean"), 평균변동=("속도표준편차","mean")
                ).reset_index().round(2)
                st.markdown("#### 군집별 특징")
                st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
                st.markdown("#### 어떤 도로가 어느 그룹일까?")
                st.dataframe(cluster_base[["_도로명","군집","평균속도","중앙속도","속도표준편차","기록수"]].sort_values(["군집","평균속도"], ascending=[True,False]).round(2),
                             use_container_width=True, hide_index=True)
                st.download_button("⬇️ 군집 결과 CSV 저장", cluster_base.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "도로_군집분석.csv", "text/csv")
                st.caption("PCA 1·2는 여러 속도 특징을 2차원으로 압축한 축이라 실제 도로 위치 좌표가 아니에요. 군집 번호 자체에 우열은 없고, 군집별 평균 속도와 변동을 읽어 의미를 붙여야 해요.")
    except ImportError:
        st.error("군집 분석에 필요한 scikit-learn이 설치되지 않았어요. requirements.txt를 다시 설치해 주세요.")

# -------------------- 탭 6 원본 --------------------
with tabs[5]:
    st.subheader("🧾 데이터 확인하기")
    st.write(f"현재 필터 결과 **{len(view):,}행 × {view.shape[1]:,}열**")
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    st.caption("화면에는 최대 500행을 미리 보여줘요. 전체 데이터는 아래 버튼으로 저장할 수 있어요.")
    st.download_button("⬇️ 현재 분석 데이터 CSV 저장", view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "분석_데이터.csv", "text/csv")
    missing_df = pd.DataFrame({
        "컬럼": raw_df.columns,
        "결측치 수": [int(raw_df[c].isna().sum()) for c in raw_df.columns],
        "결측 비율(%)": [round(float(raw_df[c].isna().mean()*100),2) for c in raw_df.columns],
        "자료형": [str(raw_df[c].dtype) for c in raw_df.columns],
        "고유값 수": [int(raw_df[c].nunique(dropna=True)) for c in raw_df.columns],
    })
    st.markdown("#### 원본 컬럼 점검")
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ 컬럼 점검표 CSV 저장", missing_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "컬럼_점검표.csv", "text/csv")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8293a3;font-size:.85rem'>🚦 도로 소통정보 분석실 · 데이터에서 패턴을 발견하고, 그래프로 설명해 봐요.</div>", unsafe_allow_html=True)
