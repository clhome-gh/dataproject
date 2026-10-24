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
    with right:
        st.write("**결측치가 있는 열**")
        missing_by_col = df.isna().sum()
        missing_table = pd.DataFrame({
            "열 이름": missing_by_col.index,
            "결측치 개수": missing_by_col.values,
            "결측 비율(%)": (missing_by_col.values / len(df) * 100).round(2),
        })
        missing_table = missing_table[missing_table["결측치 개수"] > 0]
        if missing_table.empty:
            st.success("결측치가 없습니다.")
        else:
            st.dataframe(
                missing_table.sort_values("결측치 개수", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
    st.write("**숫자형 변수 간 상관계수**")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        st.dataframe(corr.round(3), use_container_width=True)
    else:
        st.caption("상관계수를 계산할 수 있는 수치형 열이 2개 이상 필요합니다.")

with tab_columns:
    st.subheader("열별 데이터 정보")
    col_info = pd.DataFrame({
        "열 이름": df.columns,
        "자료형": [str(df[col].dtype) for col in df.columns],
        "결측치 개수": [int(df[col].isna().sum()) for col in df.columns],
        "결측 비율(%)": [(df[col].isna().mean() * 100).round(2) for col in df.columns],
        "고유값 개수": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)
    st.download_button(
        "열별 정보 CSV 다운로드",
        data=col_info.to_csv(index=False).encode("utf-8-sig"),
        file_name="column_summary.csv",
        mime="text/csv",
    )

with tab_numeric:
    st.subheader("수치형 변수의 기초통계")
    if not numeric_cols:
        st.warning("이 CSV에는 숫자형으로 인식된 열이 없습니다.")
    else:
        stats = df[numeric_cols].describe().T
        stats.insert(0, "결측치", df[numeric_cols].isna().sum())
        stats = stats.rename(columns={
            "count": "유효 개수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "25%": "1사분위수(Q1)",
            "50%": "중앙값",
            "75%": "3사분위수(Q3)",
            "max": "최댓값",
        })
        st.dataframe(stats.round(3), use_container_width=True)
        st.download_button(
            "기초통계 CSV 다운로드",
            data=stats.to_csv().encode("utf-8-sig"),
            file_name="numeric_statistics.csv",
            mime="text/csv",
        )
        st.caption("표준편차가 크면 해당 변수의 값들이 평균 주변에 넓게 퍼져 있을 가능성이 있습니다.")

with tab_categorical:
    st.subheader("범주형 변수의 빈도")
    if not non_numeric_cols:
        st.info("범주형으로 인식된 열이 없습니다.")
    else:
        selected_cat = st.selectbox("분석할 열 선택", non_numeric_cols)
        top_n = st.slider("표시할 범주 수", min_value=5, max_value=30, value=10, step=5)
        counts = df[selected_cat].fillna("(결측치)").astype(str).value_counts().head(top_n)
        freq_df = counts.rename_axis("값").reset_index(name="개수")
        freq_df["비율(%)"] = (freq_df["개수"] / len(df) * 100).round(2)
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        st.bar_chart(freq_df.set_index("값")["개수"])

with tab_chart:
    st.subheader("수치형 변수 분포")
    if not numeric_cols:
        st.info("그래프를 그릴 수 있는 수치형 열이 없습니다.")
    else:
        selected_num = st.selectbox("그래프로 확인할 수치형 열", numeric_cols, key="chart_numeric")
        values = pd.to_numeric(df[selected_num], errors="coerce").dropna()
        if values.empty:
            st.warning("선택한 열에 유효한 숫자 데이터가 없습니다.")
        else:
            st.write(f"**{selected_num}** 분포")
            fig, ax = plt.subplots()
            ax.hist(values, bins=min(30, max(5, int(values.nunique() ** 0.5))), edgecolor="black")
            ax.set_xlabel(selected_num)
            ax.set_ylabel("빈도")
            ax.set_title(f"{selected_num} 분포")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            q1, med, q3 = values.quantile([0.25, 0.5, 0.75])
            st.write(
                f"평균 **{values.mean():,.3f}** · 중앙값 **{med:,.3f}** · "
                f"표준편차 **{values.std():,.3f}** · Q1 **{q1:,.3f}** · Q3 **{q3:,.3f}**"
            )

with tab_data:
    st.subheader("데이터 미리보기")
    st.caption("아래 표는 일부 행을 보여 줍니다. 원본 CSV는 변경되지 않습니다.")
    max_rows = st.slider("미리 볼 행 수", min_value=5, max_value=100, value=20, step=5)
    st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "현재 데이터 CSV 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_preview_export.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "주의: 기초통계는 데이터의 분포를 요약합니다. 결측치가 없다고 해서 모든 값이 정확하다는 뜻은 아니며, "
    "이상치 여부는 변수의 의미와 데이터 수집 방식을 함께 고려해 판단해야 합니다."
)
