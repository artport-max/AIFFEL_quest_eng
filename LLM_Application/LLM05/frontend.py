import streamlit as st
import requests

# frontend.py 상단에 추가
def check_server():
    try:
        # 서버의 헬스체크 주소나 기본 주소에 요청을 보내봅니다.
        response = requests.get("http://127.0.0.1:8000/")
        return True
    except:
        return False

# 화면 왼쪽 사이드바나 상단에 표시
st.sidebar.markdown("### ⚙️ 설정")
if check_server():
    st.sidebar.success("● 서버 연결됨")
else:
    st.sidebar.error("○ 서버 연결 끊김")
st.set_page_config(page_title="캘리포니아 주택 가격 예측", page_icon="🏠")

st.title("🏠 캘리포니아 주택 가격 예측 서비스")
st.write("주택 정보를 입력하면 AI가 예상 가격을 실시간으로 분석합니다.")

# 1. 화면 분할 (여기서부터 수정)
col1, col2 = st.columns([1.2, 1]) # 왼쪽을 살짝 더 넓게 설정

with col1:
    st.subheader("📝 주택 정보 입력")
    # 사이드바(st.sidebar) 대신 메인 화면(st)에 배치합니다.
    med_inc = st.number_input("중위 소득 (MedInc)", value=3.0)
    house_age = st.number_input("주택 연령 (HouseAge)", value=20.0)
    ave_rooms = st.number_input("평균 방 개수 (AveRooms)", value=5.0)
    ave_bedrms = st.number_input("평균 침실 개수 (AveBedrms)", value=1.0)
    population = st.number_input("지역 인구 (Population)", value=500.0)
    ave_occup = st.number_input("평균 가구원 수 (AveOccup)", value=3.0)
    latitude = st.number_input("위도 (Latitude)", value=34.0)
    longitude = st.number_input("경도 (Longitude)", value=-118.0)

with col2:
    st.subheader("📊 예측 결과")
    st.write("") # 간격 조절용
    if st.button("AI 예측 실행", use_container_width=True): # 버튼을 꽉 차게
        payload = {
            "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
            "Latitude": latitude, "Longitude": longitude
        }
        
        with st.spinner('AI 분석 중...'):
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
        if response.status_code == 200:
            result = response.json()["predicted_price"]
            # 금액 형식으로 크게 출력
            st.metric(label="🏠 예상 주택 가격", value=f"${result*100000:,.0f}")
            st.balloons()
        else:
            st.error("서버 응답 오류")