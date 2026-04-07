from fastapi import Header, HTTPException

def get_api_key(api_key: str = Header(...)):
    # 테스트용 키입니다. 나중에 원하는 키로 바꾸세요.
    if api_key != "mysecret123": 
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key