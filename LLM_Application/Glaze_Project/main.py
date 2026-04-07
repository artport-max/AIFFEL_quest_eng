import asyncio
import sys
import os
from fastapi import FastAPI, Depends, HTTPException
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from transformers import pipeline
from auth import get_api_key  # 기존 auth.py 사용

# ── RAG 모듈 경로 추가 ──────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.glaze_rag import GlazeRAG

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# ── 1. 모델 로드 ────────────────────────────────────────
print("AI 모델을 로딩 중입니다. 잠시만 기다려 주세요...")
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn", device="cpu")
print("✅ 모델 로딩 완료!")

# ── 2. RAG 로드 ─────────────────────────────────────────
print("🔍 유약 지식 DB(RAG)를 로딩 중입니다...")
try:
    rag = GlazeRAG()
    print("✅ RAG 로딩 완료!")
except Exception as e:
    print(f"⚠️  RAG 로딩 실패 (RAG 없이 실행됩니다): {e}")
    rag = None

# ── 3. 입력 스키마 ──────────────────────────────────────
class PredictInput(BaseModel):
    text: str
    umf: dict = {}            # 유약 UMF 성분 (선택 입력)
    atmosphere: str = "oxidation"  # 소성 분위기: oxidation / reduction

# ── 4. 예측 함수 ────────────────────────────────────────
def sync_model_predict(text: str, umf: dict = {}, atmosphere: str = "oxidation"):

    # RAG 컨텍스트 생성
    rag_context = ""
    if rag is not None:
        try:
            if umf:
                # UMF 데이터가 있으면 → 성분별 상세 컨텍스트 생성
                rag_context = rag.build_analysis_context(umf, atmosphere)
            else:
                # 텍스트만 있으면 → 시맨틱 검색으로 관련 지식 검색
                results = rag.search(text, n=3)
                rag_context = "\n".join([
                    f"[{r['term']}] {r['text'][:200]}"
                    for r in results
                ])
        except Exception as e:
            rag_context = ""
            print(f"RAG 검색 오류: {e}")

    # 모델 입력 구성 (RAG 컨텍스트 + 원본 텍스트)
    if rag_context:
        enriched_text = (
            f"[유약 전문 지식]\n{rag_context}\n\n"
            f"[분석 요청]\n{text}"
        )
    else:
        enriched_text = text

    # AI 모델 실행
    result = summarizer(
        enriched_text,
        max_length=150,
        min_length=30,
        do_sample=False
    )
    return result[0].get(
        'generated_text',
        result[0].get('summary_text', '결과 생성 실패')
    )

# ── 5. 비동기 래퍼 ──────────────────────────────────────
async def run_prediction(text: str, umf: dict, atmosphere: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        sync_model_predict,
        text, umf, atmosphere
    )
    return result

# ── 6. 엔드포인트 ───────────────────────────────────────

# 기존 엔드포인트 (하위 호환 유지)
@app.post("/predict")
async def predict(input_data: PredictInput, api_key: str = Depends(get_api_key)):
    try:
        prediction = await run_prediction(
            input_data.text,
            input_data.umf,
            input_data.atmosphere
        )
        return {"status": "success", "result": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 유약 성분 직접 조회 엔드포인트 (신규)
@app.get("/glaze/oxide/{oxide_name}")
async def get_oxide_info(oxide_name: str, api_key: str = Depends(get_api_key)):
    """
    산화물 정보 직접 조회
    예: GET /glaze/oxide/Fe2O3
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG 서비스 미초기화")
    info = rag.get_oxide(oxide_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"{oxide_name} 정보 없음")
    return {"oxide": oxide_name, "info": info}


# 결함 진단 엔드포인트 (신규)
@app.get("/glaze/defect/{defect_name}")
async def get_defect_info(defect_name: str, api_key: str = Depends(get_api_key)):
    """
    유약 결함 진단 정보 조회
    예: GET /glaze/defect/Glaze Crazing
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG 서비스 미초기화")
    info = rag.diagnose_defect(defect_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"{defect_name} 정보 없음")
    return {"defect": defect_name, "info": info}


# 위험 스코어 엔드포인트 (신규)
@app.post("/glaze/risk")
async def get_risk_score(input_data: PredictInput, api_key: str = Depends(get_api_key)):
    """
    UMF 기반 결함 위험도 평가
    예: POST /glaze/risk  body: {"text": "", "umf": {"SiO2": 3.5, "Al2O3": 0.35, ...}}
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG 서비스 미초기화")
    if not input_data.umf:
        raise HTTPException(status_code=400, detail="umf 데이터가 필요합니다")
    score = rag.risk_score(input_data.umf)
    return {"status": "success", "risk": score}


# 서버 상태 확인 (신규)
@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "rag": "active" if rag is not None else "inactive",
        "model": "facebook/bart-large-cnn"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
