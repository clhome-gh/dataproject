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

    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        for sep in [None, ",", "\t", ";"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip"
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass

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


# -------------------- 시간 처리 --------------------
def parse_hour(value):
    s = str(value).strip()

    # '~1시'는 00~01시 구간이므로 0시로 변환
    # '~2시'는 01~02시 구간이므로 1시로 변환
    # ...
    # '~24시'는 23~24시 구간이므로 23시로 변환
    m = re.search(r"~\s*(\d{1,2})\s*시", s)
    if m:
        h = int(m.group(1))
        if 1 <= h <= 24:
            return h - 1
        return np.nan

    # 물결표가 없는 '01시', '2시' 등은 표시된 시각 그대로 해석
    m = re.search(r"(\d{1,2})\s*시", s)
    if m:
        h = int(m.group(1))
        return 0 if h == 24 else h

    # '1:00' 등의 형식
    m = re.search(r"(\d{1,2})\s*:\s*\d{2}", s)
    if m:
        return int(m.group(1)) % 24

    # 날짜와 시간이 함께 있는 형식
    try:
        dt = pd.to_datetime(s, errors="raise")
        return int(dt.hour)
    except Exception:
        return np.nan


def numericize(series):
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("km/h", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


uploaded = st.sidebar.file_uploader(
    "📂 1시간 소통정보 CSV 올리기",
    type=["csv", "txt"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 분석 안내")
st.sidebar.caption(
    "파일을 올리면 컬럼을 자동으로 찾아 분석해요. "
    "자동 인식이 잘못되면 아래에서 직접 선택할 수 있어요."
)
st.sidebar.caption(
    "추천: 도로명, 날짜, 시간대, 평균속도(또는 소통속도)가 들어 있는 CSV"
)

if uploaded is None:
    st.info(
        "👋 먼저 **`1시간 소통정보 (도로별-방향별).csv`** "
        "파일을 왼쪽에서 올려 주세요. 파일을 올리면 분석 화면이 열려요."
    )

    st.markdown("### 이 앱에서 볼 수 있는 것")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        '<div class="mini"><h3>📊 기초통계</h3>'
        '<p>평균·중앙값·최솟값·최댓값·표준편차·사분위수</p></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        '<div class="mini"><h3>🕒 시간대 분석</h3>'
        '<p>새벽·출근·낮·퇴근·밤의 속도 차이</p></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        '<div class="mini"><h3>📈 회귀 분석</h3>'
        '<p>시간과 속도의 관계, 추세선, R² 확인</p></div>',
        unsafe_allow_html=True
    )
    c4.markdown(
        '<div class="mini"><h3>🧩 군집 분석</h3>'
        '<p>속도 패턴이 비슷한 도로를 자동으로 묶기</p></div>',
        unsafe_allow_html=True
    )

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

time_guess = guess_col(
    cols, ["시간", "시각", "hour", "time", "기준일시", "일시"]
)
date_guess = guess_col(
    cols, ["날짜", "일자", "date", "기준일"]
)
road_guess = guess_col(
    cols, ["도로명", "도로", "노선명", "road", "구간명"]
)
speed_guess = guess_col(
    cols, ["평균속도", "속도", "speed", "소통속도", "통행속도"]
)
type_guess = guess_col(
    cols, ["도로유형", "도로종류", "도로구분", "유형", "type"]
)
direction_guess = guess_col(
    cols, ["방향", "상하행", "진행방향"]
)

st.sidebar.markdown("### 🔧 컬럼 확인")


def choose(label, guess, allow_none=True):
    opts = (["선택 안 함"] if allow_none else []) + cols

    default = (
        guess if guess in cols
        else ("선택 안 함" if allow_none else cols[0])
    )

    idx = opts.index(default) if default in opts else 0
    val = st.sidebar.selectbox(label, opts, index=idx)

    return None if val == "선택 안 함" else val


road_col = choose("도로명 컬럼", road_guess)
date_col = choose("날짜 컬럼", date_guess)
time_col = choose("시간 컬럼", time_guess)
speed_col = choose("속도 컬럼", speed_guess, allow_none=False)
type_col = choose("도로 유형 컬럼", type_guess)
direction_col = choose("방향 컬럼", direction_guess)


df = raw_df.copy()

df["_속도"] = numericize(df[speed_col])
df["_속도"] = df["_속도"].replace([np.inf, -np.inf], np.nan)
df = df[df["_속도"].notna()].copy()

if df.empty:
    st.error(
        f"'{speed_col}' 컬럼에서 숫자 속도를 찾지 못했어요. "
        "속도 컬럼을 다시 선택해 주세요."
    )
    st.stop()


if date_col:
    df["_날짜"] = pd.to_datetime(df[date_col], errors="coerce")
else:
    df["_날짜"] = pd.NaT


if time_col:
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df["_시"] = pd.to_datetime(
            df[time_col], errors="coerce"
        ).dt.hour
    else:
        parsed = df[time_col].map(parse_hour)

        if parsed.isna().mean() > 0.6:
            parsed_dt = pd.to_datetime(
                df[time_col], errors="coerce"
            )
            parsed = parsed.fillna(parsed_dt.dt.hour)

        df["_시"] = parsed
else:
    df["_시"] = np.nan


# -------------------- 시간대 분류 --------------------
def time_group(h):
    if pd.isna(h):
        return "시간 정보 없음"

    h = int(h) % 24

    if 0 <= h <= 5:
        return "🌙 새벽 (00–05)"
    if 6 <= h <= 9:
        return "🚗 출근 시간 (06–09)"
    if 10 <= h <= 16:
        return "☀️ 낮 시간 (10–16)"
    if 17 <= h <= 19:
        return "🚦 퇴근 시간 (17–19)"

    return "🌆 밤 시간 (20–23)"


df["_시간대"] = df["_시"].map(time_group)
df["_도로명"] = df[road_col].astype(str) if road_col else "도로명 정보 없음"
df["_도로유형"] = df[type_col].astype(str) if type_col else "도로 유형 정보 없음"
df["_방향"] = df[direction_col].astype(str) if direction_col else "방향 정보 없음"


if time_col:
    st.markdown(
        f'<span class="pill">시간 컬럼: {time_col}</span>'
        f'<span class="pill">속도 컬럼: {speed_col}</span>'
        f'<span class="pill">유효 속도 데이터: {len(df):,}개</span>',
        unsafe_allow_html=True
    )


# -------------------- 필터 --------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 데이터 골라보기")

roads = sorted(df["_도로명"].dropna().unique().tolist())

if len(roads) > 1 and road_col:
    selected_roads = st.sidebar.multiselect(
        "도로 선택 (비우면 전체)", roads, default=[]
    )
else:
    selected_roads = []

types = sorted(df["_도로유형"].dropna().unique().tolist())

selected_types = st.sidebar.multiselect(
    "도로 유형 선택 (비우면 전체)", types, default=[]
)

groups = [
    "🌙 새벽 (00–05)",
    "🚗 출근 시간 (06–09)",
    "☀️ 낮 시간 (10–16)",
    "🚦 퇴근 시간 (17–19)",
    "🌆 밤 시간 (20–23)",
    "시간 정보 없음"
]

selected_groups = st.sidebar.multiselect(
    "시간대 선택 (비우면 전체)", groups, default=[]
)

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

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("🚘 분석한 기록", f"{n:,}개", f"원본 {len(raw_df):,}개 행")
c2.metric("평균 속도", f"{mean_v:,.2f}", "속도 단위는 원본 데이터 기준")
c3.metric("중앙값", f"{median_v:,.2f}", "절반의 값이 이보다 작거나 같아요")
c4.metric(
    "표준편차",
    f"{std_v:,.2f}" if pd.notna(std_v) else "계산 불가",
    "값이 클수록 속도 차이가 큼"
)
c5.metric("누락 셀", f"{missing:,}", "원본 전체 기준")

st.markdown(
    '<div class="section-note">💡 <b>읽는 팁:</b> '
    '평균은 전체적인 수준을, 중앙값은 가운데 값을 보여줘요. '
    '평균과 중앙값 차이가 크면 일부 매우 빠르거나 느린 기록의 영향을 '
    '받았을 수 있어요. 아래 그래프를 함께 보면서 확인해 보세요.</div>',
    unsafe_allow_html=True
)


tabs = st.tabs([
    "📌 요약 & 기초통계",
    "🕒 시간대 패턴",
    "🛣️ 도로별 비교",
    "📈 회귀 분석",
    "🧩 군집 분석",
    "🧾 원본 데이터"
])


# -------------------- 탭 1 --------------------
with tabs[0]:
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.subheader("기초통계 요약")

        stats = view["_속도"].describe(
            percentiles=[.05, .25, .5, .75, .95]
        ).rename({
            "count": "자료 수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "5%": "5% 백분위",
            "25%": "1사분위수(Q1)",
            "50%": "중앙값(Q2)",
            "75%": "3사분위수(Q3)",
            "95%": "95% 백분위",
            "max": "최댓값"
        })

        stats_df = stats.to_frame("속도").round(3)

        st.dataframe(stats_df, use_container_width=True)

        st.download_button(
            "⬇️ 기초통계 CSV 저장",
            stats_df.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
            "기초통계.csv",
            "text/csv"
        )

        q1, q3 = view["_속도"].quantile([.25, .75])

        st.write(
            f"**중간 50%의 범위(IQR)**: "
            f"{q1:.2f} ~ {q3:.2f} (폭 {q3-q1:.2f})"
        )

        if mean_v > median_v:
            st.caption(
                "평균이 중앙값보다 높아요. 일부 높은 속도 기록이 "
                "평균을 끌어올렸을 가능성이 있어요."
            )
        elif mean_v < median_v:
            st.caption(
                "평균이 중앙값보다 낮아요. 일부 낮은 속도 기록이 "
                "평균을 끌어내렸을 가능성이 있어요."
            )
        else:
            st.caption(
                "평균과 중앙값이 비슷해요. 다만 이것만으로 분포가 "
                "완전히 대칭이라고 단정할 수는 없어요."
            )

    with right:
        st.subheader("속도 분포 살펴보기")

        fig = px.histogram(
            view,
            x="_속도",
            nbins=35,
            marginal="box",
            labels={"_속도": "속도", "count": "기록 수"},
            title="속도는 어느 구간에 가장 많이 모여 있을까?",
            color_discrete_sequence=["#2c8c99"]
        )

        fig.update_layout(
            template="plotly_white",
            height=410,
            bargap=.06,
            xaxis_title="속도",
            yaxis_title="기록 수"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()

        fig2.add_trace(
            go.Box(
                x=view["_속도"],
                name="속도 분포",
                boxmean="sd",
                marker_color="#e78b52"
            )
        )

        fig2.update_layout(
            template="plotly_white",
            title="상자수염그림으로 보는 퍼짐과 이상치",
            xaxis_title="속도",
            height=250
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("숫자형 컬럼 상관관계")

    numeric_cols = [
        c for c in view.columns
        if pd.api.types.is_numeric_dtype(view[c]) and c != "_속도"
    ]

    corr_options = list(dict.fromkeys(["_속도"] + numeric_cols))

    if len(corr_options) >= 2:
        corr = view[corr_options].corr(numeric_only=True)

        figc = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="상관계수 히트맵 (-1에 가까움: 음의 관계 / +1에 가까움: 양의 관계)"
        )

        figc.update_layout(height=450)
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info(
            "상관관계를 계산할 다른 숫자형 컬럼이 없어요. "
            "시간대 분석과 회귀 분석을 이용해 보세요."
        )


# -------------------- 탭 2 --------------------
with tabs[1]:
    st.subheader("시간대별 평균 속도")

    if view["_시"].notna().any():
        hourly = (
            view.dropna(subset=["_시"])
            .groupby("_시")["_속도"]
            .agg(["mean", "median", "count", "std"])
            .reset_index()
        )

        hourly["_시"] = hourly["_시"].astype(int)
        hourly = hourly.sort_values("_시")
        hourly["시간"] = hourly["_시"].map(lambda h: f"{h:02d}시")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=hourly["시간"],
                y=hourly["mean"],
                mode="lines+markers",
                name="평균 속도",
                line=dict(color="#247f91", width=3),
                marker=dict(size=7)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=hourly["시간"],
                y=hourly["median"],
                mode="lines+markers",
                name="중앙값",
                line=dict(color="#e78b52", width=2, dash="dot")
            )
        )

        fig.update_layout(
            template="plotly_white",
            title="시간이 바뀌면 속도도 달라질까?",
            xaxis_title="시간",
            yaxis_title="속도",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        group_order = [
            "🌙 새벽 (00–05)",
            "🚗 출근 시간 (06–09)",
            "☀️ 낮 시간 (10–16)",
            "🚦 퇴근 시간 (17–19)",
            "🌆 밤 시간 (20–23)",
            "시간 정보 없음"
        ]

        by_group = (
            view.groupby("_시간대")["_속도"]
            .agg(
                평균="mean",
                중앙값="median",
                기록수="count",
                표준편차="std"
            )
            .reindex(group_order)
            .dropna(how="all")
            .reset_index()
        )

        figg = px.bar(
            by_group,
            x="_시간대",
            y="평균",
            color="평균",
            color_continuous_scale="Teal",
            text=by_group["평균"].map(lambda x: f"{x:.1f}"),
            title="시간대별 평균 속도 비교",
            hover_data={
                "중앙값": ":.2f",
                "기록수": True,
                "표준편차": ":.2f"
            }
        )

        figg.update_layout(
            template="plotly_white",
            xaxis_title="시간대",
            yaxis_title="평균 속도",
            coloraxis_showscale=False,
            height=410
        )

        st.plotly_chart(figg, use_container_width=True)
        st.dataframe(
            by_group.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "※ '~1시'는 00~01시 구간, '~2시'는 01~02시 구간으로 "
            "해석했어요. 따라서 '~24시'는 23~24시 구간으로 처리해요."
        )

        st.subheader("시간별 기록 수")

        count_fig = px.bar(
            hourly,
            x="시간",
            y="count",
            title="시간별 데이터 기록 수",
            labels={"count": "기록 수", "시간": "시간"},
            color="count",
            color_continuous_scale="Blues"
        )

        count_fig.update_layout(
            template="plotly_white",
            height=350,
            coloraxis_showscale=False
        )

        st.plotly_chart(count_fig, use_container_width=True)

    else:
        st.warning(
            "시간 컬럼을 인식하지 못했어요. "
            "왼쪽 사이드바에서 시간 컬럼을 선택해 주세요."
        )


# -------------------- 탭 3 --------------------
with tabs[2]:
    st.subheader("도로별 속도 비교")

    if road_col:
        road_summary = (
            view.groupby("_도로명")["_속도"]
            .agg(
                평균="mean",
                중앙값="median",
                기록수="count",
                표준편차="std"
            )
            .reset_index()
        )

        min_n = st.slider(
            "도로별 비교에 포함할 최소 기록 수",
            1,
            max(2, int(road_summary["기록수"].max())),
            min(10, int(road_summary["기록수"].max()))
        )

        road_summary = road_summary[
            road_summary["기록수"] >= min_n
        ].sort_values("평균", ascending=False)

        if len(road_summary):
            max_top = min(50, len(road_summary))
            min_top = min(5, max_top)
            default_top = min(20, max_top)

            top_n = st.slider(
                "표시할 도로 수",
                min_top,
                max_top,
                default_top
            )

            low = road_summary.nsmallest(
                top_n, "평균"
            ).sort_values("평균")

            high = road_summary.nlargest(
                top_n, "평균"
            ).sort_values("평균", ascending=False)

            a, b = st.columns(2)

            with a:
                fig = px.bar(
                    high,
                    x="평균",
                    y="_도로명",
                    orientation="h",
                    text=high["평균"].map(lambda x: f"{x:.1f}"),
                    title=f"평균 속도가 높은 도로 TOP {top_n}",
                    color="평균",
                    color_continuous_scale="Teal"
                )

                fig.update_layout(
                    template="plotly_white",
                    height=max(400, top_n * 24),
                    yaxis_title="",
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with b:
                fig = px.bar(
                    low,
                    x="평균",
                    y="_도로명",
                    orientation="h",
                    text=low["평균"].map(lambda x: f"{x:.1f}"),
                    title=f"평균 속도가 낮은 도로 TOP {top_n}",
                    color="평균",
                    color_continuous_scale="Oranges"
                )

                fig.update_layout(
                    template="plotly_white",
                    height=max(400, top_n * 24),
                    yaxis_title="",
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                road_summary.sort_values(
                    "평균", ascending=False
                ).round(2),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ 도로별 통계 저장",
                road_summary.to_csv(
                    index=False,
                    encoding="utf-8-sig"
                ).encode("utf-8-sig"),
                "도로별_속도통계.csv",
                "text/csv"
            )

        else:
            st.info("최소 기록 수 조건에 맞는 도로가 없어요.")

    else:
        st.info("도로명 컬럼을 선택하면 도로별 비교가 가능해요.")

    if type_col:
        st.subheader("도로 유형별 평균 속도")

        type_summary = (
            view.groupby("_도로유형")["_속도"]
            .agg(평균="mean", 중앙값="median", 기록수="count")
            .reset_index()
            .sort_values("평균", ascending=False)
        )

        figt = px.bar(
            type_summary,
            x="_도로유형",
            y="평균",
            color="평균",
            text=type_summary["평균"].map(lambda x: f"{x:.1f}"),
            title="도로 유형별 평균 속도",
            color_continuous_scale="Sunset"
        )

        figt.update_layout(
            template="plotly_white",
            xaxis_title="도로 유형",
            yaxis_title="평균 속도",
            coloraxis_showscale=False,
            height=370
        )

        st.plotly_chart(figt, use_container_width=True)
        st.dataframe(
            type_summary.round(2),
            use_container_width=True,
            hide_index=True
        )


# -------------------- 탭 4 회귀 --------------------
with tabs[3]:
    st.subheader("📈 회귀 분석: 시간과 속도에 관계가 있을까?")

    st.markdown(
        '<div class="section-note">회귀 분석은 한 변수로 다른 변수의 '
        '경향을 설명해 보는 방법이에요. 여기서는 <b>시간(시각)</b>과 '
        '<b>속도</b>의 관계를 직선으로 살펴봐요. 관계가 있다고 해서 '
        '시간만이 속도의 원인이라고 단정할 수는 없어요.</div>',
        unsafe_allow_html=True
    )

    if (
        view["_시"].notna().sum() >= 3
        and view["_시"].nunique() >= 2
    ):
        reg = view.dropna(
            subset=["_시", "_속도"]
        )[["_시", "_속도"]].copy()

        x = reg["_시"].astype(float).to_numpy()
        y = reg["_속도"].astype(float).to_numpy()

        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept

        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        mae = float(np.mean(np.abs(y - pred)))

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "회귀식 기울기",
            f"{slope:.3f}",
            "시간 1시간 증가 시 예측 속도의 변화량"
        )
        c2.metric(
            "결정계수 R²",
            f"{r2:.3f}",
            "직선이 속도 차이를 설명하는 정도"
        )
        c3.metric(
            "평균 절대 오차(MAE)",
            f"{mae:.2f}",
            "실제값과 예측값의 평균 차이"
        )

        hourly_reg = (
            reg.groupby("_시")["_속도"]
            .mean()
            .reset_index()
        )

        xx = np.linspace(
            hourly_reg["_시"].min(),
            hourly_reg["_시"].max(),
            100
        )

        sample_reg = reg.sample(
            min(4000, len(reg)),
            random_state=4
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sample_reg["_시"],
                y=sample_reg["_속도"],
                mode="markers",
                name="개별 기록(최대 4,000개)",
                marker=dict(
                    color="#9db7c7",
                    size=4,
                    opacity=.3
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=hourly_reg["_시"],
                y=hourly_reg["_속도"],
                mode="markers+lines",
                name="시간별 평균",
                marker=dict(color="#e78b52", size=8),
                line=dict(color="#e78b52", width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xx,
                y=slope * xx + intercept,
                mode="lines",
                name="선형 회귀선",
                line=dict(color="#1c7185", width=4)
            )
        )

        fig.update_layout(
            template="plotly_white",
            title="시간(시각)과 속도의 산점도 및 회귀선",
            xaxis_title="시간(0~23시)",
            yaxis_title="속도",
            height=480
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"**회귀식:** 속도 = {intercept:.3f} "
            f"+ ({slope:.3f} × 시간)"
        )

        st.caption(
            "R²가 1에 가까울수록 직선 모형이 데이터 변동을 잘 설명하고, "
            "0에 가까울수록 설명력이 낮아요. 하루 시간은 23시 다음에 "
            "0시로 이어지는 순환형 값이라 단순 직선 회귀에는 한계가 있어요."
        )

    else:
        st.warning(
            "회귀 분석을 하려면 시간 컬럼을 선택하고, "
            "서로 다른 시간 값이 2개 이상 있어야 해요."
        )


# -------------------- 탭 5 군집 --------------------
with tabs[4]:
    st.subheader("🧩 군집 분석: 비슷한 도로끼리 묶어 보기")

    st.markdown(
        '<div class="section-note">군집 분석은 정답 라벨 없이 특징이 '
        '비슷한 대상을 자동으로 묶어요. 여기서는 도로별 평균 속도·'
        '중앙값·속도 변동을 이용해 비슷한 속도 특성을 가진 도로 그룹을 '
        '찾아요.</div>',
        unsafe_allow_html=True
    )

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        if not road_col:
            st.warning(
                "도로명 컬럼을 선택해야 도로별 군집을 만들 수 있어요."
            )

        else:
            cluster_base = (
                view.groupby("_도로명")
                .agg(
                    평균속도=("_속도", "mean"),
                    중앙속도=("_속도", "median"),
                    속도표준편차=("_속도", "std"),
                    기록수=("_속도", "count")
                )
                .reset_index()
                .fillna(0)
            )

            cluster_base = cluster_base[
                cluster_base["기록수"] >= 2
            ].copy()

            if len(cluster_base) < 3:
                st.warning(
                    "군집 분석에는 기록이 있는 도로가 최소 3개 이상 필요해요."
                )

            else:
                k = st.slider(
                    "군집 수 (K)",
                    2,
                    min(8, len(cluster_base) - 1),
                    min(3, len(cluster_base) - 1),
                    key="cluster_k"
                )

                features = [
                    "평균속도",
                    "중앙속도",
                    "속도표준편차"
                ]

                scaler = StandardScaler()
                X = scaler.fit_transform(cluster_base[features])

                model = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                cluster_base["군집"] = model.fit_predict(X) + 1

                pca = PCA(
                    n_components=2,
                    random_state=42
                )

                pcs = pca.fit_transform(X)

                cluster_base["PCA 1"] = pcs[:, 0]
                cluster_base["PCA 2"] = pcs[:, 1]

                fig = px.scatter(
                    cluster_base,
                    x="PCA 1",
                    y="PCA 2",
                    color=cluster_base["군집"].astype(str),
                    hover_name="_도로명",
                    hover_data=features + ["기록수"],
                    title="도로별 속도 특성 군집 지도(2차원으로 압축해 표시)",
                    labels={"color": "군집"},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

                fig.update_layout(
                    template="plotly_white",
                    height=500,
                    legend_title="군집"
                )

                st.plotly_chart(fig, use_container_width=True)

                cluster_summary = (
                    cluster_base.groupby("군집")
                    .agg(
                        도로수=("_도로명", "count"),
                        평균속도=("평균속도", "mean"),
                        중앙속도=("중앙속도", "mean"),
                        평균변동=("속도표준편차", "mean")
                    )
                    .reset_index()
                    .round(2)
                )

                st.markdown("#### 군집별 특징")

                st.dataframe(
                    cluster_summary,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("#### 어떤 도로가 어느 그룹일까?")

                st.dataframe(
                    cluster_base[
                        [
                            "_도로명",
                            "군집",
                            "평균속도",
                            "중앙속도",
                            "속도표준편차",
                            "기록수"
                        ]
                    ].sort_values(
                        ["군집", "평균속도"],
                        ascending=[True, False]
                    ).round(2),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ 군집 결과 CSV 저장",
                    cluster_base.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    ).encode("utf-8-sig"),
                    "도로_군집분석.csv",
                    "text/csv"
                )

                st.caption(
                    "PCA 1·2는 여러 속도 특징을 2차원으로 압축한 축이라 "
                    "실제 도로 위치 좌표가 아니에요. 군집 번호 자체에 "
                    "우열은 없고, 군집별 평균 속도와 변동을 읽어 의미를 "
                    "붙여야 해요."
                )

    except ImportError:
        st.error(
            "군집 분석에 필요한 scikit-learn이 설치되지 않았어요. "
            "requirements.txt를 다시 설치해 주세요."
        )


# -------------------- 탭 6 원본 --------------------
with tabs[5]:
    st.subheader("🧾 데이터 확인하기")

    st.write(
        f"현재 필터 결과 **{len(view):,}행 × {view.shape[1]:,}열**"
    )

    st.dataframe(
        view.head(500),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "화면에는 최대 500행을 미리 보여줘요. "
        "전체 데이터는 아래 버튼으로 저장할 수 있어요."
    )

    st.download_button(
        "⬇️ 현재 분석 데이터 CSV 저장",
        view.to_csv(
            index=False,
            encoding="utf-8-sig"
        ).encode("utf-8-sig"),
        "분석_데이터.csv",
        "text/csv"
    )

    missing_df = pd.DataFrame({
        "컬럼": raw_df.columns,
        "결측치 수": [
            int(raw_df[c].isna().sum())
            for c in raw_df.columns
        ],
        "결측 비율(%)": [
            round(float(raw_df[c].isna().mean() * 100), 2)
            for c in raw_df.columns
        ],
        "자료형": [
            str(raw_df[c].dtype)
            for c in raw_df.columns
        ],
        "고유값 수": [
            int(raw_df[c].nunique(dropna=True))
            for c in raw_df.columns
        ]
    })

    st.markdown("#### 원본 컬럼 점검")

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ 컬럼 점검표 CSV 저장",
        missing_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        ).encode("utf-8-sig"),
        "컬럼_점검표.csv",
        "text/csv"
    )


st.markdown("---")

st.markdown(
    "<div style='text-align:center;color:#8293a3;font-size:.85rem'>"
    "🚦 도로 소통정보 분석실 · 데이터에서 패턴을 발견하고, "
    "그래프로 설명해 봐요.</div>",
    unsafe_allow_html=True
)
