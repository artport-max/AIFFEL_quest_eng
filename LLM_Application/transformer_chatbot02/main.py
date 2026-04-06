import streamlit as st
import requests

st.set_page_config(page_title="Art-Trace AI 통합 대시보드", layout="wide")
st.title("🎨 Art-Trace: 통합 인공지능 시스템")

# 5. API Key 설정 (보안)
API_URL = "http://127.0.0.1:8000/chat"
headers = {"X-API-KEY": "murim_2026"} # server.py에 설정한 키와 일치해야 함

st.sidebar.header("시스템 상태")
st.sidebar.success("백엔드 서버 연결됨 (Port: 8000)")

# 채팅 인터페이스 (4.1)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("미술 비평이나 도자 예술에 대해 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # 6.2 API 레벨 통합 테스트: UI -> FastAPI 서버 요청
        with st.spinner("AI 서버가 분석 중입니다..."):
            response = requests.post(API_URL, json={"text": prompt}, headers=headers, timeout=10)
            
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
            else:
                # 5. 에러 처리
                st.error(f"인증 오류 또는 서버 에러 (Code: {response.status_code})")
    except Exception as e:
        st.error(f"서버 연결 실패: {e}")