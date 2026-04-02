import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. 데이터셋 로드
housing = fetch_california_housing()

# 2. 데이터프레임 생성 (Price라는 열을 명시적으로 추가합니다)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target  # <--- 이 부분이 'Price'라는 이름을 만드는 핵심입니다!

# 3. 상관계수 확인
print("--- 주택 가격과 다른 요소들의 상관관계 ---")
# 소득(MedInc)이 가격(Price)과 얼마나 밀접한지 보여줍니다.
correlations = df.corr()['Price'].sort_values(ascending=False)
print(correlations)

import torch
import torch.nn as nn

# 1. 모델 구조 정의 (신경망 설계도)
class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        # 입력 특징 8개 -> 은닉층 32개 -> 출력 1개
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),       # 활성화 함수 (비선형성 추가)
            nn.Linear(32, 1) # 최종 가격 1개 예측
        )
        
    def forward(self, x):
        return self.net(x)

# 2. 모델 생성 확인
model = HousingModel()
print("\n--- 생성된 PyTorch 모델 구조 ---")
print(model)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 학습용/테스트용 데이터 분리 (8:2 비율)
X = df.drop('Price', axis=1).values
y = df['Price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 데이터 스케일링 (값의 범위를 맞춰줍니다)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. PyTorch 텐서로 변환 (모델이 먹을 수 있는 형태)
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)

# 4. 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 최적화 도구
criterion = nn.MSELoss() # 손실 함수 (예측과 정답의 차이 측정)

# 5. 반복 학습 (100번만 돌려봅시다)
print("\n--- 학습 시작 ---")
for epoch in range(101):
    prediction = model(X_train_t)
    loss = criterion(prediction, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    
    # 1. 모델을 평가 모드로 전환
model.eval()

# 2. 테스트 데이터로 예측 수행
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    test_prediction = model(X_test_t)
    test_loss = criterion(test_prediction, y_test_t)

print(f"\n--- 최종 테스트 오차(MSE): {test_loss.item():.4f} ---")

# 3. 실제 값 vs 예측 값 5개만 비교해보기
print("\n[실제 가격] vs [AI 예측 가격]")
for i in range(5):
    real = y_test[i][0]
    pred = test_prediction[i][0].item()
    print(f"실제: ${real:.2f} | 예측: ${pred:.2f} (차이: {abs(real-pred):.2f})")


import joblib

# 1. 학습된 PyTorch 모델 가중치 저장
torch.save(model.state_dict(), "housing_model.pth")

# 2. 전처리에 사용한 스케일러 저장 (매우 중요!)
joblib.dump(scaler, "scaler.pkl")

print("\n--- 모델과 스케일러 저장 완료! ---")
print("파일명: housing_model.pth, scaler.pkl")

def predict_price(input_data):
    # 저장된 모델과 스케일러 불러오기 (실제 서비스 환경 가정)
    loaded_scaler = joblib.load("scaler.pkl")
    
    new_model = HousingModel()
    new_model.load_state_dict(torch.load("housing_model.pth"))
    new_model.eval()
    
    # 데이터 전처리 및 예측
    scaled_data = loaded_scaler.transform([input_data])
    input_tensor = torch.FloatTensor(scaled_data)
    
    with torch.no_grad():
        prediction = new_model(input_tensor)
    
    return prediction.item()

# 예시 데이터로 한 번 테스트 (첫 번째 특징: MedInc 소득 3.0인 경우)
sample_input = [3.0, 15.0, 5.0, 1.0, 300.0, 3.0, 37.0, -122.0]
result = predict_price(sample_input)
print(f"\n테스트 입력에 대한 예측 가격: ${result:.2f} ($100,000 단위)")


import joblib

# 1. 모델 가중치 저장 (.pth)
torch.save(model.state_dict(), "housing_model.pth")

# 2. 전처리 파라미터(스케일러) 저장 (.pkl)
joblib.dump(scaler, "scaler.pkl")

print("✅ 모델 및 전처리 파라미터 저장 완료!")

import torch
import torch.nn as nn
import joblib
import numpy as np

# 학습 때와 동일한 모델 구조
class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 모델과 스케일러를 불러와 예측하는 클래스
class Predictor:
    def __init__(self, model_path="housing_model.pth", scaler_path="scaler.pkl"):
        self.scaler = joblib.load(scaler_path)
        self.model = HousingModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, input_list):
        # 정규화 -> 텐서 변환 -> 추론
        scaled_data = self.scaler.transform([input_list])
        input_tensor = torch.FloatTensor(scaled_data)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.item()

if __name__ == "__main__":
    predictor = Predictor()
    # 임의의 테스트 데이터 (MedInc: 3.0, HouseAge: 15.0 등 8개)
    test_data = [3.0, 15.0, 5.0, 1.0, 300.0, 3.0, 37.0, -122.0]
    result = predictor.predict(test_data)
    print(f"🔍 테스트 데이터 예측 결과: ${result:.2f}")


import torch
import torch.nn as nn
import joblib

# 1. 모델 구조 (설계도)
class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. 추론 클래스 (도구를 사용하는 사람)
class Predictor:
    def __init__(self, model_path="housing_model.pth", scaler_path="scaler.pkl"):
        self.scaler = joblib.load(scaler_path)
        self.model = HousingModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, input_list):
        scaled_data = self.scaler.transform([input_list])
        input_tensor = torch.FloatTensor(scaled_data)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.item()
    

from fastapi import FastAPI
from pydantic import BaseModel
from inference import Predictor  # 위에서 만든 모듈을 가져옵니다!

app = FastAPI()
predictor = Predictor()

# Pydantic 스키마: 데이터 형식을 정의 (교재 필수 과정)
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
    # 입력 데이터를 리스트로 변환하여 추론 모듈에 전달
    input_list = [
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]
    prediction = predictor.predict(input_list)
    return {"predicted_price": round(prediction, 4)}