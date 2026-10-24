import io
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CSV 기초통계 분석기",
    page_icon="📊",
    layout="wide",
)

st.title("📊 CSV 데이터 기초통계 분석기")
st.write(
    "CSV 파일을 업로드하면 행·열 개수, 결측치, 중복 행, 중앙값, 평균, 표준편차, "
    "범주별 빈도와 그래프를 확인할 수 있습니다."
)

with st.sidebar:
    st.header("데이터 불러오기")
    uploaded_file = st.file_uploader("분석할 CSV 파일을 선택하세요", type=["csv"])
    st.caption("UTF-8, UTF-8-SIG, CP949, EUC-KR 인코딩을 순서대로 확인합니다.")

def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """여러 인코딩과 구분자를 순서대로 시도해 CSV를 읽습니다."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            # sep=None과 python 엔진을 사용해 쉼표/탭/세미콜론 구분자를 추정합니다.
            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="error",
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV를 읽을 수 없습니다. 파일 형식과 인코딩을 확인해 주세요.\n" + "\n".join(errors))

if uploaded_file is None:
    st.info("왼쪽에서 CSV 파일을 업로드하면 분석이 시작됩니다.")
    st.subheader("분석할 수 있는 항목")
    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 크기", "행 × 열")
    c2.metric("데이터 품질", "결측치 · 중복 행")
    c3.metric("기초통계", "평균 · 중앙값 · 사분위수")
    st.markdown(
        """
        **사용 방법**
        1. `Browse files`를 눌러 CSV 파일을 선택합니다.
        2. 요약 탭에서 데이터의 전체 크기와 결측치를 확인합니다.
        3. 수치형 통계와 범주형 통계 탭에서 변수별 분포를 살펴봅니다.
        """
    )
    st.stop()

try:
    df = load_csv(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"파일을 불러오지 못했습니다: {exc}")
    st.stop()

# 열 이름의 앞뒤 공백은 제거하되 원본 데이터 값은 그대로 둡니다.
df.columns = [str(col).strip() for col in df.columns]

if df.empty:
    st.warning("파일에 데이터 행이 없습니다.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
missing_total = int(df.isna().sum().sum())
duplicate_total = int(df.duplicated().sum())
memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

# 상단 핵심 지표
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("행 개수", f"{len(df):,}")
m2.metric("열 개수", f"{df.shape[1]:,}")
m3.metric("결측 셀", f"{missing_total:,}")
m4.metric("중복 행", f"{duplicate_total:,}")
m5.metric("메모리 사용량", f"{memory_mb:.2f} MB")

tab_overview, tab_columns, tab_numeric, tab_categorical, tab_chart, tab_data = st.tabs(
    ["전체 요약", "열별 정보", "수치형 기초통계", "범주형 통계", "그래프", "데이터 미리보기"]
)

with tab_overview:
    st.subheader("데이터 요약")
    left, right = st.columns(2)
    with left:
        st.write("**데이터 유형별 열 개수**")
        dtype_counts = df.dtypes.astype(str).value_counts().rename_axis("자료형").reset_index(name="열 개수")
        st.dataframe(dtype_counts, use_container_width=True, hide_index=True)
