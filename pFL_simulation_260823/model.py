import torch 
import torch.nn as nn

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


# 3. 개인화 연합학습 (pFL) 메인 모델
class PFLHealthModel(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32, hidden_dim=64):
        super().__init__()
        
        # [글로벌 공유 특성 추출기] (MobileNetV3 스타일 Backbone)
        # 여러 클라이언트 간 공유되어 일반적인 건강 지표 패턴을 학습
        self.shared_extractor = nn.Sequential(
            # Input Stem
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Hardswish(inplace=True),
            
            # Inverted Bottleneck Block 1
            TabularInvertedResidual(embed_dim, hidden_dim, embed_dim),
            # Inverted Bottleneck Block 2
            TabularInvertedResidual(embed_dim, hidden_dim, embed_dim),
        )

        # [로컬 개인화 헤드]
        # 사용자 고유의 생체 반응 특성에 맞게 로컬에서만 학습 및 유지
        self.local_head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Tanh() # 출력 범위: -1.0 ~ +1.0
        )

    def forward(self, x):
        features = self.shared_extractor(x)
        out = self.local_head(features)
        return out
