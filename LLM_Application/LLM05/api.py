from fastapi import FastAPI
from pydantic import BaseModel
from inference import Predictor # inference.py에서 Predictor를 가져옵니다

app = FastAPI()
predictor = Predictor()

@app.get("/")
def health_check():
    return {"status": "ok"}

# Pydantic 스키마 (교재 내용)
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def get_prediction(data: HousingInput):
    input_list = [
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]
    prediction = predictor.predict(input_list)
    return {"predicted_price": round(prediction, 4)}

