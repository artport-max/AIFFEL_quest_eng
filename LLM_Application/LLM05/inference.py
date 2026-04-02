import torch
import torch.nn as nn
import joblib

# 모델 구조 정의
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

# 추론용 클래스
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