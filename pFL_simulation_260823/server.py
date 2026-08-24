import flwr as fl
# 총 3대의 클라이언트가 모두 연결되어야 각 라운드 학습 진행
NUM_CLIENTS = 3
NUM_ROUNDS = 20
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, # 접속된 모든 클라이언트 참여
    min_fit_clients=NUM_CLIENTS, # 최소 학습 참가 클라이언트 수
    min_available_clients=NUM_CLIENTS, # 최소 접속 대기 클라이언트 수
)

if __name__ == "__main__":
    print(f"=========================================================")
    print(f" [Server] PFL Aggregation Server Started on Port 8080 ")
    print(f" [Server] Waiting for {NUM_CLIENTS} distributed clients to connect... ")
    print(f"=========================================================")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )