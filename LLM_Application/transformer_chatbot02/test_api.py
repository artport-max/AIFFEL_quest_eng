import requests
import time

# 박무림 님 서버 설정과 일치시켜야 합니다
API_BASE = "http://localhost:8000"
VALID_KEY = "murim_2026"  # server.py의 API_KEY와 동일하게!

headers = {"X-API-KEY": VALID_KEY}

print("\n" + "="*60)
print("🚀 6.2 API 레벨 통합 테스트 시작")
print("="*60)

# 테스트 1: 인증 및 기본 추론 테스트
def test_basic_inference():
    print("[테스트 1] 기본 추론 및 인증 확인...")
    payload = {"text": "도자 예술의 역사에 대해 알려주세요."}
    
    try:
        start_time = time.time()
        resp = requests.post(f"{API_BASE}/chat", json=payload, headers=headers)
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            answer = resp.json().get("answer", "")
            print(f"✅ 성공! (소요시간: {elapsed:.2f}초)")
            print(f"🤖 AI 응답: {answer[:100]}...\n")
        else:
            print(f"❌ 실패! 에러 코드: {resp.status_code}\n")
    except Exception as e:
        print(f"❌ 서버 연결 불가: {e}\n")

# 테스트 2: 멀티턴 시뮬레이션
def test_multiturn():
    print("[테스트 2] 멀티턴 시뮬레이션...")
    questions = ["청자란 무엇인가?", "그것의 특징은?"]
    
    for q in questions:
        print(f"👤 질문: {q}")
        resp = requests.post(f"{API_BASE}/chat", json={"text": q}, headers=headers)
        if resp.status_code == 200:
            print(f"🤖 답변: {resp.json()['answer'][:50]}...")
        else:
            print(f"❌ 에러 발생: {resp.status_code}")
    print("\n")

# 실행
if __name__ == "__main__":
    test_basic_inference()
    test_multiturn()
    print("="*60)
    print("🎉 모든 API 레벨 테스트 종료")
    print("="*60)


# 테스트 3: 입력 검증 (에러 핸들링 확인)
def test_validation():
    print("[테스트 3] 입력 검증 및 에러 핸들링...")
    
    # 1. 잘못된 API Key 테스트
    wrong_headers = {"X-API-KEY": "wrong_key_123"}
    resp = requests.post(f"{API_BASE}/chat", json={"text": "안녕"}, headers=wrong_headers)
    print(f"🚩 잘못된 키 테스트 -> HTTP {resp.status_code} (403이면 성공)")

    # 2. 빈 메시지 전송 테스트
    resp = requests.post(f"{API_BASE}/chat", json={"text": ""}, headers=headers)
    print(f"🚩 빈 메시지 테스트 -> HTTP {resp.status_code} (200 혹은 422 예상)")
    print("\n")

# 테스트 4: 동시 요청 (병렬 처리 확인)
from concurrent.futures import ThreadPoolExecutor

def send_chat(i):
    resp = requests.post(f"{API_BASE}/chat", json={"text": f"질문 {i}"}, headers=headers)
    return resp.status_code

def test_concurrency():
    print("[테스트 4] 동시 요청 테스트 (4개 병렬)...")
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(send_chat, range(4)))
    
    end = time.time()
    print(f"✅ 동시 요청 완료: {results}")
    print(f"⏱️ 총 소요 시간: {end - start:.2f}초")

# 메인 실행부 수정
if __name__ == "__main__":
    test_basic_inference()
    test_multiturn()
    test_validation()  # 추가
    test_concurrency() # 추가
    print("="*60)
    print("🎉 모든 API 레벨 테스트 진짜 종료!")
    print("="*60)