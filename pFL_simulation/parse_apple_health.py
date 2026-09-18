import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# 1. 파일 경로 설정 (압축 해제 후 생성된 export.xml 경로)
XML_PATH = "apple_health_export/export.xml"  # 본인의 export.xml 경로로 수정

print(">> 애플 건강 데이터 파싱을 시작합니다. (대용량 스트리밍 처리 중...)")

# 데이터 저장용 딕셔너리 (날짜별)
sleep_data = defaultdict(float)      # {date: total_sleep_hours}
rhr_data = defaultdict(list)         # {date: [rhr_values]}
hrv_data = defaultdict(list)         # {date: [hrv_values]}
calories_data = defaultdict(float)   # {date: total_active_calories}

# XML 파싱 (iterparse로 메모리 절약)
context = ET.iterparse(XML_PATH, events=("end",))

for event, elem in context:
    if elem.tag == "Record":
        record_type = elem.attrib.get("type", "")
        start_date_str = elem.attrib.get("startDate", "")
        end_date_str = elem.attrib.get("endDate", "")
        value = elem.attrib.get("value", "")

        try:
            # 1) 수면 데이터 (HKCategoryTypeIdentifierSleepAnalysis)
            if record_type == "HKCategoryTypeIdentifierSleepAnalysis":
                # 수면 상태 (Asleep Core/Deep/REM 등) 체크
                # 'HKCategoryValueSleepAnalysisInBed'를 제외한 실제 수면 단계 합산
                if "Asleep" in value or value in ["1", "3", "4", "5"]:
                    start_dt = datetime.strptime(start_date_str[:19], "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(end_date_str[:19], "%Y-%m-%d %H:%M:%S")
                    duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
                    
                    # 기상한 날짜(end_dt의 날짜)를 기준으로 수면 시간 배정
                    wake_date = end_dt.strftime("%Y-%m-%d")
                    sleep_data[wake_date] += duration_hours

            # 2) 안정시 심박수 (HKQuantityTypeIdentifierRestingHeartRate)
            elif record_type == "HKQuantityTypeIdentifierRestingHeartRate":
                date = start_date_str[:10]
                rhr_data[date].append(float(value))

            # 3) 심박변이도 HRV (HKQuantityTypeIdentifierHeartRateVariabilitySDNN)
            elif record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
                date = start_date_str[:10]
                hrv_data[date].append(float(value))

            # 4) 활성 소모 칼로리 (HKQuantityTypeIdentifierActiveEnergyBurned)
            elif record_type == "HKQuantityTypeIdentifierActiveEnergyBurned":
                date = start_date_str[:10]
                calories_data[date] += float(value)

        except Exception:
            pass

        # 메모리 정리 (300MB 대용량 처리 핵심)
        elem.clear()

print(">> XML 파싱 완료. 일별 데이터 정렬 및 결합 중...")

# 모든 날짜 수집
all_dates = sorted(list(set(list(sleep_data.keys()) + list(rhr_data.keys()) + list(hrv_data.keys()) + list(calories_data.keys()))))

records = []
for date in all_dates:
    # 당일 수면, RHR, HRV
    sleep = sleep_data.get(date, np.nan)
    rhr = np.mean(rhr_data[date]) if date in rhr_data and len(rhr_data[date]) > 0 else np.nan
    hrv = np.mean(hrv_data[date]) if date in hrv_data and len(hrv_data[date]) > 0 else np.nan
    
    # "전날(Yesterday)"의 활성 소모 칼로리 매핑
    yesterday = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_cal = calories_data.get(yesterday, np.nan)

    # 4가지 피처가 모두 존재하는 날만 추출
    if not (np.isnan(sleep) or np.isnan(rhr) or np.isnan(hrv) or np.isnan(prev_cal)):
        # 비정상적인 극단값 필터링 (e.g. 수면 2시간 미만 / 14시간 이상 등)
        if 2.0 <= sleep <= 14.0:
            records.append({
                "date": date,
                "sleep_hours": round(sleep, 2),
                "rhr": round(rhr, 1),
                "hrv": round(hrv, 1),
                "active_calories": round(prev_cal, 1)
            })

df = pd.DataFrame(records)

# -------------------------------------------------------------
# 5) Label (target_ratio) 생성 로직
# -------------------------------------------------------------
# 실제 사용자의 주관적 라벨링이 없다면, 생리학적 회복 점수 기반 초기 라벨 생성
# (Z-score 기반: 수면/HRV 높으면 +, RHR/칼로리 높으면 -)
z_sleep = (df["sleep_hours"] - df["sleep_hours"].mean()) / df["sleep_hours"].std()
z_hrv = (df["hrv"] - df["hrv"].mean()) / df["hrv"].std()
z_rhr = (df["rhr"] - df["rhr"].mean()) / df["rhr"].std()
z_cal = (df["active_calories"] - df["active_calories"].mean()) / df["active_calories"].std()

# 컨디션 인덱스 = (회복 지표) - (부하 지표)
condition_score = (0.35 * z_sleep + 0.35 * z_hrv) - (0.15 * z_rhr + 0.15 * z_cal)
df["target_ratio"] = np.clip(np.tanh(condition_score * 0.5), -0.8, 0.5).round(2)

# CSV 저장 (학습용 포맷)
output_file = "health_data.csv"
df.to_csv(output_file, index=False)

print(f">> 성공적으로 {len(df)}일치 데이터가 추출되어 '{output_file}'로 저장되었습니다!")
print(df.head(10))