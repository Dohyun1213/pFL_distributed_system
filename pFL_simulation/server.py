import flwr as fl
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

import sys

# 총 9대의 클라이언트가 모두 연결되어야 각 라운드 학습 진행 (기존 3대 + 데스크탑 시뮬레이션 6대)
NUM_CLIENTS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 9
NUM_ROUNDS = 100

# 1. 클라이언트 메트릭 가중 평균 집계 함수
def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}
    
    weighted_acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    weighted_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
    weighted_mae = sum(num_examples * m.get("mae", 0.0) for num_examples, m in metrics) / total_examples
    total_bytes = sum(m.get("upload_bytes", 0) for _, m in metrics)

    return {
        "accuracy": weighted_acc,
        "loss": weighted_loss,
        "mae": weighted_mae,
        "total_upload_bytes": total_bytes,
    }

# 2. 정확도 및 데이터 수신 현황을 출력하는 커스텀 FedAvg 전략
class PFLServerStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if results:
            print(f"\n" + "="*65)
            print(f" [Server Round {server_round}] 글로벌 모델 집계 및 성능 요약")
            print(f" ---------------------------------------------------------------")
            print(f" ▶ 참여 클라이언트 수 : {len(results)}대 (실패: {len(failures)}대)")
            
            # 각 클라이언트별 개별 정확도 및 손실 출력
            for idx, (_, res) in enumerate(results, 1):
                c_acc = res.metrics.get("accuracy", 0.0)
                c_loss = res.metrics.get("loss", 0.0)
                c_mae = res.metrics.get("mae", 0.0)
                print(f"   - 클라이언트 #{idx}: 정확도 {c_acc:.2f}% | MSE: {c_loss:.4f} | MAE: {c_mae:.4f}")

            # 전체 클라이언트 가중 평균 지표 출력
            if aggregated_metrics:
                print(f" ---------------------------------------------------------------")
                if "accuracy" in aggregated_metrics:
                    print(f" ▶ 글로벌 평균 정확도 (Averaged Accuracy) : {aggregated_metrics['accuracy']:.2f}%")
                if "loss" in aggregated_metrics:
                    print(f" ▶ 글로벌 가중 평균 손실 (MSE / MAE) : {aggregated_metrics['loss']:.4f} / {aggregated_metrics.get('mae', 0.0):.4f}")
                if "total_upload_bytes" in aggregated_metrics:
                    total_b = aggregated_metrics["total_upload_bytes"]
                    print(f" ▶ 총 수신 데이터 용량 : {total_b/1024.0:.2f} KB ({total_b:,} Bytes)")
            print(f"="*65 + "\n")

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if results and aggregated_metrics and "accuracy" in aggregated_metrics:
            print(f" >> [Server Round {server_round} Eval] 글로벌 평가 정확도: {aggregated_metrics['accuracy']:.2f}% | Loss: {loss:.4f}")
            
        return loss, aggregated_metrics

strategy = PFLServerStrategy(
    fraction_fit=1.0, # 접속된 모든 클라이언트 참여
    min_fit_clients=NUM_CLIENTS, # 최소 학습 참가 클라이언트 수
    min_available_clients=NUM_CLIENTS, # 최소 접속 대기 클라이언트 수
    fit_metrics_aggregation_fn=weighted_average_metrics,
    evaluate_metrics_aggregation_fn=weighted_average_metrics,
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