import streamlit as st
import json
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── 1. 페이지 설정 ──────────────────────────────────────
st.set_page_config(
    page_title="ArtGlaze Cloud — 유약 분석",
    layout="wide",
    page_icon="🏺"
)
st.title("🏺 ArtGlaze Cloud: AI 유약 분석 시스템")
st.info("유약 레시피를 선택하면 AI가 화학적 안정성 · 발색 특징 · 예술적 질감을 분석합니다.")

API_URL  = "http://127.0.0.1:8000"
API_KEY  = "mysecret123"
HEADERS  = {"api-key": API_KEY}

# ── 2. 서버 상태 확인 ───────────────────────────────────
@st.cache_data(ttl=10)
def check_server():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

server_status = check_server()
if server_status:
    rag_status = "🟢 활성" if server_status.get("rag") == "active" else "🔴 비활성"
    st.sidebar.success(f"서버 연결됨  |  RAG: {rag_status}")
else:
    st.sidebar.error("⚠️ 서버 미연결 — main.py를 먼저 실행하세요")

# ── 3. 데이터 로드 ──────────────────────────────────────
DATA_PATH = 'data/sampled_glaze/combined_glaze.json'

if not os.path.exists(DATA_PATH):
    st.error("데이터 파일을 찾을 수 없습니다. preprocess.py를 먼저 실행해주세요.")
    st.stop()

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    glaze_data = json.load(f)

glaze_names = [item['name'] for item in glaze_data]

# ── 4. 사이드바 — 유약 선택 ─────────────────────────────
st.sidebar.header("⚙️ 분석 설정")
selected_name = st.sidebar.selectbox("분석할 유약 선택:", glaze_names)
atmosphere    = st.sidebar.radio("소성 분위기:", ["oxidation", "reduction"], index=0,
                                  format_func=lambda x: "산화 소성" if x == "oxidation" else "환원 소성")

selected_item = next(item for item in glaze_data if item['name'] == selected_name)

# ── 5. UMF 성분 파싱 ────────────────────────────────────
# combined_glaze.json 안에 성분 데이터가 있으면 사용,
# 없으면 input_for_ai 텍스트에서 파싱 시도
def extract_umf(item: dict) -> dict:
    """JSON에서 UMF 성분 추출. 없으면 빈 dict 반환."""
    # 키 이름 후보들 확인
    for key in ['umf', 'UMF', 'composition', 'oxides', 'recipe']:
        if key in item:
            return item[key]
    # 직접 산화물 키가 있는 경우
    oxide_keys = ['SiO2', 'Al2O3', 'CaO', 'MgO', 'K2O', 'Na2O',
                  'Fe2O3', 'ZnO', 'TiO2', 'B2O3', 'CoO', 'CuO']
    found = {k: item[k] for k in oxide_keys if k in item}
    return found

umf = extract_umf(selected_item)

# ── 6. 메인 화면 — 2컬럼 레이아웃 ──────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"📊 {selected_name} — 성분 분석")

    if umf:
        # 실제 UMF 데이터로 차트
        df_umf = pd.DataFrame({
            '산화물': list(umf.keys()),
            '함량':   list(umf.values())
        }).sort_values('함량', ascending=False)

        fig_pie = px.pie(
            df_umf, values='함량', names='산화물', hole=0.35,
            color_discrete_sequence=px.colors.sequential.RdBu,
            title="UMF 산화물 비율"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # 성분 표
        st.dataframe(
            df_umf.style.highlight_max(axis=0, color='#fde2e2'),
            use_container_width=True,
            hide_index=True
        )

    else:
        # UMF 데이터 없을 때 — 임시 mock 데이터 (기존 유지)
        st.caption("⚠️ UMF 데이터 없음 — 임시 데이터 표시")
        mock_data = pd.DataFrame(dict(
            value=[60, 15, 10, 5, 10],
            variable=['Silica', 'Alumina', 'Lime', 'Magnesia', 'Others']
        ))
        fig = px.pie(mock_data, values='value', names='variable', hole=.3,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            display_df = mock_data.copy()
            display_df.columns = ['함량(%)', '화학 성분명']
            display_df = display_df[['화학 성분명', '함량(%)']]
            st.dataframe(
                display_df.style.highlight_max(axis=0, color='#fde2e2'),
                use_container_width=True, hide_index=True
            )
        with col2:
            main_comp = display_df.iloc[display_df['함량(%)'].idxmax()]
            st.metric(label="주요 성분", value=main_comp['화학 성분명'],
                      delta=f"{main_comp['함량(%)']}%")
            st.caption("비중이 가장 높은 성분입니다.")

with col_right:
    st.subheader(f"🔍 {selected_name} — 기본 정보")
    st.text_area("데이터 원문:", selected_item.get('input_for_ai', ''), height=120)

    # ── RAG 위험 스코어 (UMF 있을 때만) ──────────────
    if umf and server_status:
        st.markdown("---")
        st.subheader("⚠️ 결함 위험 스코어")
        try:
            r = requests.post(
                f"{API_URL}/glaze/risk",
                json={"text": "", "umf": umf, "atmosphere": atmosphere},
                headers=HEADERS, timeout=5
            )
            if r.status_code == 200:
                risk = r.json()["risk"]

                # 위험도 색상 매핑
                color_map = {"low": "🟢", "medium": "🟡", "high": "🔴"}

                c1, c2, c3 = st.columns(3)
                c1.metric("크레이징 위험",
                          f"{color_map.get(risk['crazing_risk'], '⚪')} {risk['crazing_risk'].upper()}")
                c2.metric("표면 예측", risk['surface_type'].upper())
                c3.metric("Si:Al 비율", risk['si_al_ratio'])

                st.caption(f"알칼리 합계(Na₂O+K₂O): {risk['alkali_total']}")
        except Exception as e:
            st.warning(f"위험 스코어 조회 실패: {e}")

# ── 7. AI 분석 실행 버튼 ────────────────────────────────
st.markdown("---")
if st.button("🤖 AI 유약 분석 실행", type="primary", use_container_width=True):

    if not server_status:
        st.error("서버가 실행되지 않았습니다. 터미널에서 main.py를 먼저 실행하세요.")
    else:
        with st.spinner("AI가 유약을 분석 중입니다... (RAG 지식 검색 포함)"):
            try:
                payload = {
                    "text": selected_item.get('input_for_ai', selected_name),
                    "umf": umf,
                    "atmosphere": atmosphere
                }
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    headers=HEADERS,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()["result"]
                    st.success("✅ AI 분석 완료")

                    # 결과 표시 — 3개 탭
                    tab1, tab2, tab3 = st.tabs(
                        ["🧪 화학적 안정성", "🎨 발색 특징", "✨ 예술적 질감"]
                    )
                    # 결과 텍스트를 3등분해서 각 탭에 표시
                    result_len = len(result)
                    chunk = max(result_len // 3, 50)

                    with tab1:
                        st.markdown("#### 화학적 안정성 분석")
                        st.write(result[:chunk] if result_len > chunk else result)
                    with tab2:
                        st.markdown("#### 발색 특징 분석")
                        st.write(result[chunk:chunk*2] if result_len > chunk*2 else result)
                    with tab3:
                        st.markdown("#### 예술적 질감 분석")
                        st.write(result[chunk*2:] if result_len > chunk*2 else result)

                    # 전체 결과도 표시
                    with st.expander("📄 전체 분석 결과 보기"):
                        st.write(result)

                else:
                    st.error(f"서버 오류: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error(f"연결 실패: {e}")
