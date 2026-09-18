import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict

# ==============================================================================
# CLIA (Class-grained Logits Interaction-based Architecture) 건강 모델
# ------------------------------------------------------------------------------
# 1. PIA vs CLIA 개념 비교:
#    - PIA (Parameters Interaction-based Architecture):
#        클라이언트와 서버가 '모델 가중치 파라미터'(shared_extractor) 전체를 교환.
#        통신량: 약 101.38 KB (103,808 Bytes) / 라운드당
#    - CLIA (Class-grained Logits Interaction-based Architecture):
#        모델 파라미터를 전혀 전송하지 않고, 각 클래스(건강 상태 등급)별 평균 '로짓(Logits)'만 교환.
#        통신량: 약 0.10 KB (100 Bytes, num_classes=5 기준) / 라운드당 -> 99.9% 통신량 절감!
# 2. 공정한 비교를 위해 PIA와 동일한 MobileNetV3 기반 Inverted Residual Backbone 유지
# ==============================================================================

# 1. MobileNetV3 스타일의 Squeeze-and-Excitation (SE) Feature Attention
class TabularSEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # 피처별 중요도 (0.0 ~ 1.0) 가중치 생성
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight


# 2. MobileNetV3 스타일의 Inverted Residual Block (MLP 버전)
class TabularInvertedResidual(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.use_residual = (in_features == out_features)

        self.block = nn.Sequential(
            # Expand (차원 확장)
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.Hardswish(inplace=True),
            
            # Squeeze-and-Excitation (중요 특성 강조)
            TabularSEBlock(hidden_features),
            
            # Project (차원 축소)
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# 3. CLIA 메인 모델
class CLIAHealthModel(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32, hidden_dim=64, num_blocks=4, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
        # [로컬 특성 추출기 Backbone] - PIA와 동일한 구조/용량으로 공정한 비교 보장
        # CLIA에서는 이 Backbone 파라미터를 서버로 절대 전송하지 않고 로컬에만 유지합니다.
        extractor_layers = [
            # Input Stem
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Hardswish(inplace=True),
        ]
        for _ in range(num_blocks):
            extractor_layers.append(
                TabularInvertedResidual(embed_dim, hidden_dim, embed_dim)
            )
        self.feature_extractor = nn.Sequential(*extractor_layers)

        # [특성 프로젝션 레이어]
        self.shared_proj = nn.Sequential(
            nn.Linear(embed_dim, 24),
            nn.LayerNorm(24),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1)
        )

        # [CLIA Logit Head] - 클래스별 로짓(Logit)을 생성하여 연합 지식 증류(KD)에 활용
        # 서버와 교환하는 로짓 상호작용의 핵심 레이어
        self.logit_head = nn.Linear(24, num_classes)

        # [로컬 회귀 예측 Head] - 연속형 건강 지표(target_ratio) 최종 예측
        self.reg_head = nn.Sequential(
            nn.Linear(24, 12),
            nn.LayerNorm(12),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(12, 1),
            nn.Tanh() # 출력 범위: -1.0 ~ +1.0
        )

    def forward(self, x, return_logits=False):
        """
        순전파:
        - return_logits=False: 회귀 예측값만 반환 (일반 추론 모드)
        - return_logits=True: (회귀 예측값, 클래스 로짓) 튜플 반환 (CLIA 학습 모드)
        """
        features = self.feature_extractor(x)
        proj = self.shared_proj(features)
        reg_out = self.reg_head(proj)
        
        if return_logits:
            logits = self.logit_head(proj)
            return reg_out, logits
        return reg_out

    def get_logits(self, x):
        """입력 데이터에 대한 클래스 로짓(Logits) 벡터만 추출"""
        features = self.feature_extractor(x)
        proj = self.shared_proj(features)
        return self.logit_head(proj)


# ==============================================================================
# CLIA 연합학습 알고리즘 보조 유틸리티 함수들
# ==============================================================================

def continuous_to_class(y: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    연속형 target_ratio (-1.0 ~ +1.0)를 num_classes개의 건강 상태 등급으로 이산화(Binning).
    예: 5등급 분류 기준 (0: 매우 저하, 1: 저하, 2: 보통, 3: 양호, 4: 최상)
    """
    if num_classes == 5:
        # [-0.3, -0.1, 0.1, 0.3] 기준 분할
        bins = torch.tensor([-0.3, -0.1, 0.1, 0.3], device=y.device, dtype=y.dtype)
        return torch.bucketize(y, bins).squeeze(-1)
    else:
        # 균등 분할
        min_val, max_val = -1.0, 1.0
        step = (max_val - min_val) / num_classes
        bins = torch.linspace(min_val + step, max_val - step, num_classes - 1, device=y.device, dtype=y.dtype)
        return torch.bucketize(y, bins).squeeze(-1)


def compute_class_grained_logits(
    model: CLIAHealthModel,
    dataloader,
    num_classes: int = 5,
    device: str = "cpu"
) -> np.ndarray:
    """
    [CLIA 핵심] 로컬 데이터셋에서 각 클래스(c)별 평균 로짓 벡터(Class-grained Logits)를 계산.
    
    반환값:
        np.ndarray: shape (num_classes, num_classes), float32
        -> 이 2차원 배열(5x5 = 100 Bytes)만이 클라이언트가 서버로 전송하는 유일한 데이터입니다!
    """
    model.eval()
    class_logits_sum = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            labels = continuous_to_class(batch_y, num_classes=num_classes)
            logits = model.get_logits(batch_x)  # shape: (batch_size, num_classes)

            for c in range(num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_logits_sum[c] += logits[mask].sum(dim=0)
                    class_counts[c] += mask.sum()

    # 클래스별 평균 계산
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_logits_sum[c] /= class_counts[c]

    return class_logits_sum.cpu().numpy().astype(np.float32)


def clia_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    global_class_logits: Optional[torch.Tensor],
    temperature: float = 2.0
) -> torch.Tensor:
    """
    서버로부터 수신한 글로벌 클래스 로짓(global_class_logits)과 로컬 예측 로짓 간의 지식 증류(KD) 손실 계산.
    KL Divergence를 통해 파라미터 전송 없이도 로컬 모델이 글로벌 클래스 지식을 흡수하도록 유도합니다.
    """
    if global_class_logits is None:
        return torch.tensor(0.0, device=logits.device)

    # 각 샘플의 정답 라벨에 해당하는 글로벌 클래스 로짓 타깃 매핑
    target_logits = global_class_logits[labels]  # shape: (batch_size, num_classes)
    
    p_s = F.log_softmax(logits / temperature, dim=1)
    p_t = F.softmax(target_logits / temperature, dim=1)
    
    kd_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
    return kd_loss


def print_communication_comparison(num_classes: int = 5):
    """PIA와 CLIA 간의 전송 통신량 비교 리포트 출력"""
    from model import PFLHealthModel
    pia_model = PFLHealthModel()
    pia_bytes = sum(val.cpu().numpy().nbytes for val in pia_model.shared_extractor.state_dict().values())
    
    clia_payload = np.zeros((num_classes, num_classes), dtype=np.float32)
    clia_bytes = clia_payload.nbytes
    
    reduction = (1 - clia_bytes / pia_bytes) * 100.0

    print("=" * 65)
    print(" [연합학습 통신량 비교] PIA vs CLIA")
    print("-" * 65)
    print(f" ▶ PIA  (Parameters Interaction): {pia_bytes / 1024.0:.2f} KB ({pia_bytes:,} Bytes)")
    print(f" ▶ CLIA (Class-grained Logits)  : {clia_bytes / 1024.0:.2f} KB ({clia_bytes:,} Bytes)")
    print(f" ▶ 통신량 절감률                : {reduction:.2f}% 절감 (약 {pia_bytes // clia_bytes}배 경량화)")
    print("=" * 65)


if __name__ == "__main__":
    print_communication_comparison()
