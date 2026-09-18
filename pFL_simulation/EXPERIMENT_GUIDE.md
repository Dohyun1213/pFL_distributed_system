# 🚀 분산 개인화 연합학습 (pFL) 9인 분산 실험 가이드

본 문서는 **총 9대의 클라이언트(물리 3대 + 데스크탑 가상 시뮬레이션 6대)**를 활용하여 **PIA(가중치 교환)**와 **CLIA(로짓 교환)** 방식의 통신량 및 모델 정확도를 비교 실험하기 위한 매뉴얼입니다.

---

## 📌 1. 실험 환경 및 참여자 구성

| 구분 | 담당자 | 역할 | 실행 위치 | 클라이언트 번호 |
| :--- | :--- | :--- | :--- | :--- |
| **서버** | **도현 (데스크탑)** | 중앙 FedAvg / Logit 집계 서버 | 데스크탑 (포트 8080 / 8081) | - |
| **클라이언트 1** | **동훈** | 원격 물리 클라이언트 1대 | 동훈 노트북/PC | Client #1 |
| **클라이언트 2** | **태인** | 원격 물리 클라이언트 1대 | 태인 노트북/PC | Client #2 |
| **클라이언트 3** | **도현** | 로컬 물리 클라이언트 1대 | 도현 데스크탑 | Client #3 |
| **클라이언트 4~9** | **도현 (가상)** | 데스크탑 시뮬레이션 클라이언트 6대 | 도현 데스크탑 (백그라운드) | Client #4 ~ #9 |

- **총 참가 클라이언트 수**: **9대** (9대가 모두 연결되어야 각 라운드 학습이 시작됩니다)
- **총 진행 라운드 수**: **100 라운드**
- **데이터셋 분할**:
  - 클라이언트 1, 2, 3: 각자의 로컬 생체 데이터 (`health_data.csv`)
  - 클라이언트 4 ~ 9: `Donghoon/health_data.csv`를 6개로 균등 분할(~98행씩)하여 독립 학습

---

## 👨‍💻 2. 동훈 & 태인 참여 가이드 (원격 클라이언트)

> 💡 **참고**: 도현이가 데스크탑에서 서버를 먼저 켠 후, 도현이가 알려준 **서버 IP 주소**를 입력하고 접속합니다.

### 1단계: 최신 코드 내려받기
터미널을 열고 프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다:
```bash
# 1. 최신 코드 pull
git pull origin main

# 2. 시뮬레이션 디렉토리로 이동
cd pFL_simulation
```

### 2단계: 가상환경 활성화
각자의 파이썬 가상환경을 활성화합니다 (예시):
```bash
source ~/leo_fl_env/bin/activate
# 또는
conda activate <가상환경이름>
```

### 3단계: 실험 실행

#### [실험 1] PIA 기반 연합학습 (가중치 교환 모드)
도현이가 PIA 서버(포트 8080)를 실행하면, 아래 명령어를 입력하여 접속합니다:
```bash
# <도현_서버_IP> 부분에 도현이가 알려준 IP(예: Tailscale IP 100.x.x.x) 입력
python client.py <도현_서버_IP>:8080
```
- 예시: `python client.py 100.85.120.45:8080`

#### [실험 2] CLIA 기반 연합학습 (초경량 로짓 교환 모드)
도현이가 CLIA 서버(포트 8081)를 실행하면, 아래 명령어를 입력하여 접속합니다:
```bash
# CLIA는 기본 포트 8081 사용
python client_CLIA.py <도현_서버_IP>:8081
```
- 예시: `python client_CLIA.py 100.85.120.45:8081`

---

## 🖥️ 3. 도현 데스크탑 실행 가이드 (서버 + 7대 클라이언트)

도현이는 데스크탑에서 **① 서버 1개**, **② 로컬 클라이언트 1개**, **③ 가상 클라이언트 6개**를 터미널 3개로 나누어 실행합니다.

### 사전 준비: 데스크탑 IP 확인
동훈이와 태인이에게 알려줄 IP를 확인합니다:
```bash
# Tailscale 환경인 경우:
tailscale ip -4
# 동일 공유기/로컬망인 경우:
hostname -I | awk '{print $1}'
```

---

### [실험 1] PIA 연합학습 실행 순서 (포트 8080)

#### 터미널 1: PIA 중앙 서버 시작
```bash
cd /home/dohyun-kim/GNU/pFL_distributed_system/pFL_simulation
source ~/leo_fl_env/bin/activate
python server.py
```
> `[Server] Waiting for 9 distributed clients to connect...` 문구가 뜨며 대기합니다.

#### 터미널 2: 도현 로컬 클라이언트 (Client #3) 실행
```bash
cd /home/dohyun-kim/GNU/pFL_distributed_system/pFL_simulation
source ~/leo_fl_env/bin/activate
python client.py 127.0.0.1:8080 3
```

#### 터미널 3: 데스크탑 가상 클라이언트 6대 (Client #4 ~ #9) 일괄 실행
```bash
cd /home/dohyun-kim/GNU/pFL_distributed_system/pFL_simulation
source ~/leo_fl_env/bin/activate
python simulate_clients.py --server 127.0.0.1:8080 --mode pia --num-clients 6
```

> 🎯 **결과**: 동훈(Client #1), 태인(Client #2), 도현 로컬(Client #3), 가상 6대(Client #4~#9) 총 9대가 연결되면 **서버가 자동으로 100 라운드 학습을 시작**합니다.

---

### [실험 2] CLIA 연합학습 실행 순서 (포트 8081)

#### 터미널 1: CLIA 중앙 서버 시작
```bash
cd /home/dohyun-kim/GNU/pFL_distributed_system/pFL_simulation
source ~/leo_fl_env/bin/activate
python server_CLIA.py 8081
```

#### 터미널 2: 도현 로컬 클라이언트 (Client #3) 실행
```bash
cd /home/dohyun-kim/GNU/pFL_distributed_system/pFL_simulation
source ~/leo_fl_env/bin/activate
python client_CLIA.py 127.0.0.1:8081 3
```

#### 터미널 3: 데스크탑 가상 클라이언트 6대 (Client #4 ~ #9) 일괄 실행
```bash
cd /home/dohyun-kim/GNU/pFL_distributed_system/pFL_simulation
source ~/leo_fl_env/bin/activate
python simulate_clients.py --server 127.0.0.1:8081 --mode clia --num-clients 6
```

---

## 📊 4. 결과 비교 관전 포인트 (PIA vs CLIA)

서버 터미널에 라운드마다 출력되는 요약 리포트를 통해 두 방식을 대조합니다:

```text
=================================================================
 [Server Round 1] 글로벌 모델 집계 및 성능 요약
 ---------------------------------------------------------------
 ▶ 참여 클라이언트 수 : 9대 (실패: 0대)
   - 클라이언트 #1 (동훈) : 정확도 97.50% | MSE: 0.0022
   - 클라이언트 #2 (태인) : 정확도 98.10% | MSE: 0.0019
   - 클라이언트 #3 (도현) : 정확도 96.80% | MSE: 0.0025
   - 클라이언트 #4 ~ #9 (시뮬레이션 6대) ...
 ---------------------------------------------------------------
 ▶ 글로벌 평균 정확도 (Averaged Accuracy) : 97.80%
 ▶ 총 수신 데이터 용량 : 
    - PIA  : 약 912.44 KB (9대 x 101.38 KB)
    - CLIA : 약 0.88 KB   (9대 x 0.10 KB) -> 99.9% 통신량 절감!
=================================================================
```

1. **통신량 절감 효과**: 라운드당 수신 용량이 900 KB대(PIA)에서 1 KB 미만(CLIA)으로 급감함을 확인
2. **정확도 수렴 추이**: 100 라운드 동안 평균 정확도(Averaged Accuracy)가 얼마나 빠르게, 높게 유지되는지 비교
3. **개별 클라이언트 적합성**: Non-IID 환경에서 각 클라이언트의 개인화 정확도가 안정적으로 상승하는지 관찰

---

## ❓ 5. 트러블슈팅 (자주 묻는 질문)

- **Q. `Connection refused` 에러가 떠요!**
  - 도현 데스크탑에서 서버(`server.py` 또는 `server_CLIA.py`)가 켜져 있는지 확인하세요.
  - IP 주소와 포트 번호(`8080` or `8081`)가 올바른지 확인하세요.
- **Q. 클라이언트를 종료하고 싶어요!**
  - 터미널에서 `Ctrl + C`를 누르면 클라이언트 프로세스들이 안전하게 종료됩니다.
