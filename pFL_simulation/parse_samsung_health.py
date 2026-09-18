import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# 1. 삼성 헬스 압축 해제 폴더 경로 (기본 폴더명 패턴 자동 탐색)
SEARCH_PATH = "./samsunghealth_*"
folders = glob.glob(SEARCH_PATH)
DATA_DIR = folders[0] if folders else "./"

print(f">> 삼성 헬스 데이터 폴더 탐색: {DATA_DIR}")

def find_file(pattern):
    """지정된 패턴의 CSV 파일 경로 검색"""
    matches = glob.glob(os.path.join(DATA_DIR, pattern))
    if not matches:
        matches = glob.glob(os.path.join(DATA_DIR, "**", pattern), recursive=True)
    return matches[0] if matches else None

# 데이터 저장소
sleep_data = defaultdict(float)
rhr_data = defaultdict(list)
hrv_data = defaultdict(list)
calories_data = defaultdict(float)

# -------------------------------------------------------------
# 1. 수면 데이터 파싱 (com.samsung.shealth.sleep.csv)
# -------------------------------------------------------------
sleep_file = find_file("*sleep.csv")
if sleep_file:
    print(f" > 수면 파일 로드: {os.path.basename(sleep_file)}")
    try:
        # 삼성 CSV는 상단 메타데이터가 있을 수 있어 1행 건너뜀 (없으면 header=0으로 자동 처리)
        df_sleep = pd.read_csv(sleep_file, skiprows=1)
        if "start_time" not in df_sleep.columns:
            df_sleep = pd.read_csv(sleep_file)

        for _, row in df_sleep.iterrows():
            start_str = str(row.get("start_time", ""))
            end_str = str(row.get("end_time", ""))
            if len(start_str) >= 19 and len(end_str) >= 19:
                start_dt = datetime.strptime(start_str[:19], "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_str[:19], "%Y-%m-%d %H:%M:%S")
                duration = (end_dt - start_dt).total_seconds() / 3600.0
                wake_date = end_dt.strftime("%Y-%m-%d")
                sleep_data[wake_date] += duration
    except Exception as e:
        print(f" [!] 수면 파싱 중 오류: {e}")

# -------------------------------------------------------------
# 2. 안정시 심박수 파싱 (com.samsung.shealth.resting_heart_rate.csv)
# -------------------------------------------------------------
rhr_file = find_file("*resting_heart_rate.csv") or find_file("*heart_rate.csv")
if rhr_file:
    print(f" > 심박수 파일 로드: {os.path.basename(rhr_file)}")
    try:
        df_rhr = pd.read_csv(rhr_file, skiprows=1)
        if "start_time" not in df_rhr.columns and "create_time" not in df_rhr.columns:
            df_rhr = pd.read_csv(rhr_file)

        time_col = "start_time" if "start_time" in df_rhr.columns else "create_time"
        val_col = "resting_heart_rate" if "resting_heart_rate" in df_rhr.columns else "heart_rate"

        for _, row in df_rhr.iterrows():
            date_str = str(row.get(time_col, ""))[:10]
            val = row.get(val_col)
            if pd.notnull(val) and len(date_str) == 10:
                rhr_data[date_str].append(float(val))
    except Exception as e:
        print(f" [!] 심박수 파싱 중 오류: {e}")

# -------------------------------------------------------------
# 3. 스트레스 / HRV 파싱 (com.samsung.shealth.stress.csv)
# -------------------------------------------------------------
stress_file = find_file("*stress.csv")
if stress_file:
    print(f" > 스트레스/HRV 파일 로드: {os.path.basename(stress_file)}")
    try:
        df_stress = pd.read_csv(stress_file, skiprows=1)
        if "start_time" not in df_stress.columns and "create_time" not in df_stress.columns:
            df_stress = pd.read_csv(stress_file)

        time_col = "start_time" if "start_time" in df_stress.columns else "create_time"
        # 삼성 헬스는 스트레스 점수(0~100) 또는 HRV(ms)를 기록함 (스트레스 낮을수록 HRV 높음)
        for _, row in df_stress.iterrows():
            date_str = str(row.get(time_col, ""))[:10]
            if "hrv" in df_stress.columns and pd.notnull(row.get("hrv")):
                hrv_data[date_str].append(float(row["hrv"]))
            elif "score" in df_stress.columns and pd.notnull(row.get("score")):
                # 스트레스 점수(0~100) 역산 근사 (스트레스 100 -> HRV 20ms, 스트레스 0 -> HRV 90ms)
                approx_hrv = 90.0 - (float(row["score"]) * 0.7)
                hrv_data[date_str].append(approx_hrv)
    except Exception as e:
        print(f" [!] 스트레스 파싱 중 오류: {e}")

# -------------------------------------------------------------
# 4. 활성 소모 칼로리 (com.samsung.shealth.activity.day_summary.csv)
# -------------------------------------------------------------
cal_file = find_file("*activity.day_summary.csv") or find_file("*calories_burned*.csv")
if cal_file:
    print(f" > 활동/칼로리 파일 로드: {os.path.basename(cal_file)}")
    try:
        df_cal = pd.read_csv(cal_file, skiprows=1)
        if "create_time" not in df_cal.columns and "day_time" not in df_cal.columns:
            df_cal = pd.read_csv(cal_file)

        time_col = "day_time" if "day_time" in df_cal.columns else ("create_time" if "create_time" in df_cal.columns else df_cal.columns[0])
        val_col = "active_calorie" if "active_calorie" in df_cal.columns else ("calorie" if "calorie" in df_cal.columns else "calories")

        for _, row in df_cal.iterrows():
            date_str = str(row.get(time_col, ""))[:10]
            val = row.get(val_col)
            if pd.notnull(val) and len(date_str) == 10:
                calories_data[date_str] = float(val)
    except Exception as e:
        print(f" [!] 칼로리 파싱 중 오류: {e}")

# -------------------------------------------------------------
# 5. 일별 데이터 결합 및 결측치 보정
# -------------------------------------------------------------
all_dates = sorted(list(set(list(sleep_data.keys()) + list(rhr_data.keys()) + list(hrv_data.keys()) + list(calories_data.keys()))))

full_records = []
for date in all_dates:
    yesterday = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    full_records.append({
        "date": date,
        "sleep_hours": sleep_data.get(date, np.nan),
        "rhr": np.mean(rhr_data[date]) if date in rhr_data and len(rhr_data[date]) > 0 else np.nan,
        "hrv": np.mean(hrv_data[date]) if date in hrv_data and len(hrv_data[date]) > 0 else np.nan,
        "active_calories": calories_data.get(yesterday, np.nan)
    })

df_full = pd.DataFrame(full_records)

# 결측치 보정 (Forward-fill + median)
df_full["rhr"] = df_full["rhr"].ffill().bfill()
df_full["hrv"] = df_full["hrv"].ffill().bfill()
df_full["active_calories"] = df_full["active_calories"].ffill().bfill()
df_full["sleep_hours"] = df_full["sleep_hours"].fillna(df_full["sleep_hours"].median())

# 유효 범위 필터링
df = df_full[(df_full["sleep_hours"] >= 2.0) & (df_full["sleep_hours"] <= 14.0)].copy()

# -------------------------------------------------------------
# 6. 라벨(target_ratio) 생성 및 저장
# -------------------------------------------------------------
z_sleep = (df["sleep_hours"] - df["sleep_hours"].mean()) / (df["sleep_hours"].std() + 1e-7)
z_hrv = (df["hrv"] - df["hrv"].mean()) / (df["hrv"].std() + 1e-7)
z_rhr = (df["rhr"] - df["rhr"].mean()) / (df["rhr"].std() + 1e-7)
z_cal = (df["active_calories"] - df["active_calories"].mean()) / (df["active_calories"].std() + 1e-7)

condition_score = (0.35 * z_sleep + 0.35 * z_hrv) - (0.15 * z_rhr + 0.15 * z_cal)
df["target_ratio"] = np.clip(np.tanh(condition_score * 0.5), -0.8, 0.5).round(2)

output_cols = ["sleep_hours", "rhr", "hrv", "active_calories", "target_ratio"]
output_file = "health_data.csv"
df[output_cols].to_csv(output_file, index=False)

print(f">> [완료] 삼성 헬스 데이터 {len(df)}일치가 '{output_file}'로 저장되었습니다.")
print(df[output_cols].head())