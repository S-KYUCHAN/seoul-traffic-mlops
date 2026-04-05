# app.py
# FastAPI + PyTorch 서빙 예제
# 포인트: 모델 로드 시점(시작 시 1번), 요청마다 inference만 수행

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import os

# ----- 입력/출력 스키마 -----
class PredictRequest(BaseModel):
    # instances: [[x1, x2, x3, x4], ...]
    instances: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

# ----- 모델 정의 (학습 때와 동일 구조 유지가 중요) -----
class TrafficMLP(nn.Module):
    def __init__(self, hidden_size=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(4, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----- 전역 객체 (서버 시작 시 1번만 로드) -----
app = FastAPI(title="Seoul Traffic MLP Serving")

MODEL_DIR = os.environ.get("MODEL_DIR", "/model/v1")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/artifact/v1")

# 예: /model/v1/model.pt, /artifact/v1/param.json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: TrafficMLP = None
Y_mean: float = 0.0
Y_std: float = 1.0

def load_model():
    global model, Y_mean, Y_std

    # 하이퍼파라미터 + 스케일링 파라미터 로드
    with open(os.path.join(ARTIFACT_DIR, "param.json")) as f:
        params = json.load(f)

    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    Y_mean = params["Y_mean"]
    Y_std = params["Y_std"]

    model = TrafficMLP(hidden_size=hidden_size, num_layers=num_layers).to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 모델 크기나 파라미터 수 정도는 로그로 찍어두면 디버깅/면접 때 설명하기 좋음
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[serving] Loaded model from {MODEL_DIR}, params={num_params}")

@app.on_event("startup")
def startup_event():
    # 서버 시작 시점에 모델 1번 로드
    load_model()

@app.get("/healthz")
def health():
    # 단순 헬스체크 엔드포인트 (K8s liveness/readiness에 활용 가능)
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) 입력 validation
    if len(req.instances) == 0:
        return PredictResponse(predictions=[])

    # 2) 텐서 변환
    x = torch.tensor(req.instances, dtype=torch.float32, device=device)
    # 3) 추론 (정규화된 y를 예측한 뒤, 원래 scale로 복원)
    with torch.no_grad():
        pred_norm = model(x).squeeze(-1)
        pred_vol = pred_norm * Y_std + Y_mean
        preds = pred_vol.cpu().tolist()

    return PredictResponse(predictions=preds)