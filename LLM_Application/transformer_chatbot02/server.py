from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Art-Trace Backend")

# 5. API Key 인증 설정 (보안)
API_KEY = "murim_2026"
api_key_header = APIKeyHeader(name="X-API-KEY")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="인증 실패")

# 모델 로드 (서버 실행 시 한 번만 로드)
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to("cuda")

# 3.2 Pydantic 스키마 (데이터 형식 정의)
class ChatRequest(BaseModel):
    text: str

# 3.3 & 3.4 API 엔드포인트 및 추론
@app.post("/chat")
async def chat(request: ChatRequest):
    input_ids = tokenizer.encode(request.text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, 
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.5
        )
    
    answer = tokenizer.decode(gen_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"answer": answer}