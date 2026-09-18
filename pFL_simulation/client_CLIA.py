import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import flwr as fl
from model_CLIA import (
    CLIAHealthModel,
    continuous_to_class,
    compute_class_grained_logits,
    clia_distillation_loss
)

# 1. 데이터 로드 및 로컬 Z-score 표준화
print("[Client-CLIA] Loading local dataset (health_data.csv)...")
df = pd.read_csv("health_data.csv")

feature_cols = ["sleep_hours", "rhr", "hrv", "active_calories"]
X_raw = df[feature_cols].values
y_raw = df[["target_ratio"]].values

# 개별 사용자 기저치 기준 정규화
X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-7)

dataset = TensorDataset(
    torch.tensor(X_norm, dtype=torch.float32),
    torch.tensor(y_raw, dtype=torch.float32)
)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. CLIA 모델 및 옵티마이저 초기화
NUM_CLASSES = 5
model = CLIAHealthModel(num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 3. Flower NumPyClient 정의 (CLIA: 파라미터 대신 클래스별 평균 로짓만 교환)
class CLIAHealthClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.round = 0
        self.global_class_logits: torch.Tensor = None

    def get_parameters(self, config):
        # [CLIA 핵심] 모델 파라미터를 보내지 않고, 로컬 클래스별 평균 로짓(100 Bytes)만 반환
        class_logits = compute_class_grained_logits(model, train_loader, num_classes=NUM_CLASSES)
        return [class_logits]

    def set_parameters(self, parameters):
        # [CLIA 핵심] 서버에서 취합된 글로벌 클래스 로짓을 수신 (로컬 모델 가중치는 보존)
        if parameters and len(parameters) > 0:
            self.global_class_logits = torch.tensor(parameters[0], dtype=torch.float32)

    def compute_metrics(self):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                all_preds.append(model(batch_x))
                all_targets.append(batch_y)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        mse = criterion(preds, targets).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        tolerance = 0.10
        accuracy = (torch.abs(preds - targets) <= tolerance).float().mean().item() * 100.0
        return mse, mae, accuracy

    def fit(self, parameters, config):
        self.round += 1
        current_round = config.get("server_round", self.round)

        self.set_parameters(parameters)
        model.train()
        
        # 로컬 학습 5 에포크 (Task Loss + 글로벌 로짓 지식 증류 KD Loss)
        for epoch in range(5):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred, logits = model(batch_x, return_logits=True)
                loss_task = criterion(pred, batch_y)
                
                # 글로벌 클래스 로짓이 수신된 경우 지식 증류(KD) 적용
                if self.global_class_logits is not None:
                    labels = continuous_to_class(batch_y, num_classes=NUM_CLASSES)
                    loss_kd = clia_distillation_loss(logits, labels, self.global_class_logits, temperature=2.0)
                    loss = loss_task + 0.5 * loss_kd
                else:
                    loss = loss_task
                    
                loss.backward()
                optimizer.step()

        # 1. 로컬 모델 정확도 및 평가 지표 계산
        mse, mae, accuracy = self.compute_metrics()

        # 2. 서버로 전송할 클래스 로짓(Class-grained Logits) 추출 및 데이터 용량 계산
        upload_params = self.get_parameters(config={})
        upload_bytes = sum(arr.nbytes for arr in upload_params)
        upload_kb = upload_bytes / 1024.0

        # 3. 라운드별 전송 용량 및 모델 정확도 출력
        print(f"\n" + "="*65)
        print(f" [CLIA Round {current_round}] 로컬 학습 및 로짓 전송 요약")
        print(f" ---------------------------------------------------------------")
        print(f" ▶ 서버 전송 데이터 용량 : {upload_kb:.2f} KB ({upload_bytes:,} Bytes) [초경량]")
        print(f" ▶ 로컬 모델 정확도 (오차 ±0.10 이내) : {accuracy:.2f}%")
        print(f" ▶ 로컬 모델 손실 (MSE / MAE) : {mse:.4f} / {mae:.4f}")
        print(f"="*65 + "\n")

        return upload_params, len(train_loader.dataset), {
            "loss": float(mse),
            "mae": float(mae),
            "accuracy": float(accuracy),
            "upload_bytes": int(upload_bytes)
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        mse, mae, accuracy = self.compute_metrics()
        current_round = config.get("server_round", self.round)
        print(f" >> [CLIA Round {current_round} Eval] Test Loss(MSE): {mse:.4f} | Accuracy: {accuracy:.2f}%")
        return float(mse), len(train_loader.dataset), {
            "loss": float(mse),
            "mae": float(mae),
            "accuracy": float(accuracy)
        }

# 4. 서버 연결 실행
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_CLIA.py <server_address>")
        print("Example: python client_CLIA.py 127.0.0.1:8081")
        sys.exit(1)

    server_address = sys.argv[1]
    print(f"[Client-CLIA] Connecting to CLIA Server at {server_address}...")
    fl.client.start_client(
        server_address=server_address,
        client=CLIAHealthClient().to_client()
    )
