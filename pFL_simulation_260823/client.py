import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import flwr as fl
from model import PFLHealthModel

# 1. 데이터 로드 및 로컬 Z-score 표준화
print("[Client] Loading local dataset (health_data.csv)...")
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

# 2. 모델 및 옵티마이저 초기화
model = PFLHealthModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 3. Flower NumPyClient 정의 (Shared Layer만 가중치 동기화)
class HealthClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Shared Extractor의 가중치만 서버로 반환
        return [val.cpu().numpy() for val in model.shared_extractor.state_dict().values()]
    
    def set_parameters(self, parameters):
        # 서버에서 수신한 가중치를 Shared Extractor에만 반영 (Local Head 보존)
        params_dict = zip(model.shared_extractor.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.shared_extractor.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        epoch_losses = []
        for epoch in range(5): # 로컬 에포크 5회
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_losses.append(total_loss / len(train_loader))
        
        print(f" >> Local Epoch Loss: {epoch_losses[-1]:.4f}")
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                pred = model(batch_x)
                loss += criterion(pred, batch_y).item()
        loss /= len(train_loader)
        return float(loss), len(train_loader.dataset), {"loss": float(loss)}

# 4. 서버 연결 실행
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py ")
        print("Example: python client.py 100.85.120.45:8080")
        sys.exit(1)

    server_address = sys.argv[1]
    print(f"[Client] Connecting to PFL Server at {server_address}...")
    fl.client.start_client(
        server_address=server_address,
        client=HealthClient().to_client()
    )