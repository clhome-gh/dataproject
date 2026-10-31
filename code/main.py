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

df["_시간코드"] = pd.to_numeric(df[time_col], errors="coerce")
valid_time_code = df["_시간코드"].between(1, 24, inclusive="both")
df["_시"] = np.where(valid_time_code, df["_시간코드"] - 1, np.nan)
df["_시"] = pd.to_numeric(df["_시"], errors="coerce")

def time_group(h):
    if pd.isna(h): 
        return "시간 정보 없음"
    h = int(h) % 24
    if 0 <= h <= 5: 
        return "🌙 새벽 (00-05)"
    if 6 <= h <= 9: 
        return "🚗 출근 시간 (06-09)"
    if 10 <= h <= 16: 
        return "☀️ 낮 시간 (10-16)"
    if 17 <= h <= 19: 
        return "🚦 퇴근 시간 (17-19)"
    return "🌆 밤 시간 (20-23)"

df["_시간대"] = df["_시"].map(time_group)
df["_도로명"] = df[road_col].astype(str) if road_col else "도로명 정보 없음"
df["_도로유형"] = df[type_col].astype(str) if type_col else "도로 유형 정보 없음"
df["_방향"] = df[direction_col].astype(str) if direction_col else "방향 정보 없음"

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
groups = ["🌙 새벽 (00-05)", "🚗 출근 시간 (06-09)", "☀️ 낮 시간 (10-16)", "🚦 퇴근 시간 (17-19)", "🌆 밤 시간 (20-23)", "시간 정보 없음"]
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
