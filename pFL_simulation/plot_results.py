import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 스타일 및 폰트 설정
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "DejaVu Sans"

# 2. 데이터 불러오기
df_pia = pd.read_csv("experiment_results_pia.csv")
df_clia = pd.read_csv("experiment_results_clia.csv")

# ==============================================================================
# [그래프 1] 종합 비교 대시보드 (4-in-1 Dashboard)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

# (1) 라운드별 글로벌 정확도 수렴 추이
ax1 = axs[0, 0]
ax1.plot(df_pia["round"], df_pia["global_avg_acc"], marker="o", markersize=6, color="#1f77b4", linewidth=2.2, label="PIA (Parameter Exchange)")
ax1.plot(df_clia["round"], df_clia["global_avg_acc"], marker="s", markersize=6, color="#ff7f0e", linewidth=2.2, linestyle="--", label="CLIA (Class-Logit Distillation)")
ax1.set_title("[1] Global Accuracy Convergence (9 Clients)", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Communication Round", fontsize=11)
ax1.set_ylabel("Global Accuracy (%)", fontsize=11)
ax1.set_ylim(92, 101)
ax1.legend(fontsize=10, loc="lower right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.6)

# (2) 1회 라운드당 통신량 비교 (로그 스케일)
ax2 = axs[0, 1]
categories = ["PIA\n(Model Parameters)", "CLIA\n(Class Logits)"]
volumes = [912.38, 0.88] # KB
colors = ["#1f77b4", "#2ca02c"]
bars = ax2.bar(categories, volumes, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
ax2.set_yscale("log")
ax2.set_title("[2] Uplink Payload per Round (9 Clients Total)", fontsize=13, fontweight="bold", pad=10)
ax2.set_ylabel("Data Transferred (KB) [Log Scale]", fontsize=11)
ax2.set_ylim(0.1, 4000)
for bar, val in zip(bars, volumes):
    ax2.text(bar.get_x() + bar.get_width()/2, val * 1.35, f"{val:.2f} KB\n({val*1024:,.0f} B)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.text(0.5, 0.72, "★ 99.90% Communication Reduction\n(1,038x Ultra-Lightweight)", transform=ax2.transAxes, ha="center", fontsize=11, fontweight="bold", color="#d62728",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebeb", edgecolor="#d62728", linewidth=1.5))

# (3) 누적 통신량 추이 (MB)
ax3 = axs[1, 0]
rounds = df_clia["round"]
cum_clia_mb = df_clia["upload_data_kb"].cumsum() / 1024.0
pia_rounds = df_pia["round"]
cum_pia_mb = df_pia["upload_data_kb"].cumsum() / 1024.0

ax3.plot(pia_rounds, cum_pia_mb, marker="o", markersize=5, color="#1f77b4", linewidth=2.2, label="PIA Cumulative (MB)")
ax3.plot(rounds, cum_clia_mb, marker="s", markersize=5, color="#2ca02c", linewidth=2.2, linestyle="--", label="CLIA Cumulative (MB)")
ax3.set_title("[3] Cumulative Network Traffic (19 Rounds)", fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Communication Round", fontsize=11)
ax3.set_ylabel("Total Transferred (MB)", fontsize=11)
ax3.legend(fontsize=10, loc="upper left", frameon=True)
ax3.grid(True, linestyle="--", alpha=0.6)

final_pia_mb = cum_pia_mb.iloc[-1]
final_clia_mb = cum_clia_mb.iloc[-1]
ax3.annotate(f"PIA: {final_pia_mb:.2f} MB", xy=(19, final_pia_mb), xytext=(13.5, final_pia_mb - 2.5),
             arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6), fontweight="bold")
ax3.annotate(f"CLIA: {final_clia_mb*1024:.1f} KB\n({final_clia_mb:.4f} MB)", xy=(19, final_clia_mb), xytext=(12, 2.5),
             arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6), fontweight="bold")

# (4) 9대 클라이언트별 최종 개인화 정확도 비교 (Round 19)
ax4 = axs[1, 1]
client_labels = [f"Client #{i}" for i in range(1, 10)]
pia_r19 = df_pia[df_pia["round"] == 19].iloc[0][[f"client_{i}_acc" for i in range(1, 10)]].values
clia_r19 = df_clia[df_clia["round"] == 19].iloc[0][[f"client_{i}_acc" for i in range(1, 10)]].values

x = np.arange(len(client_labels))
width = 0.35
ax4.bar(x - width/2, pia_r19, width, label="PIA Round 19", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.8)
ax4.bar(x + width/2, clia_r19, width, label="CLIA Round 19", color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.8)
ax4.set_title("[4] Personalization Accuracy per Client (Round 19)", fontsize=13, fontweight="bold", pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(client_labels, fontsize=9, rotation=25)
ax4.set_ylabel("Accuracy (%)", fontsize=11)
ax4.set_ylim(94, 102)
ax4.legend(fontsize=10, loc="lower right", frameon=True)
ax4.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("experiment_dashboard.png", dpi=300)
plt.close()
print("▶ [Saved] experiment_dashboard.png (종합 4-in-1 대시보드)")

# ==============================================================================
# [그래프 2] 단독 논문용: 정확도 수렴 곡선 (fig1_accuracy.png)
# ==============================================================================
plt.figure(figsize=(9, 6), dpi=300)
plt.plot(df_pia["round"], df_pia["global_avg_acc"], marker="o", markersize=7, color="#1f77b4", linewidth=2.5, label="PIA (Parameter Interaction)")
plt.plot(df_clia["round"], df_clia["global_avg_acc"], marker="s", markersize=7, color="#e377c2", linewidth=2.5, linestyle="--", label="CLIA (Class-grained Logit Interaction)")
plt.title("Convergence Comparison: PIA vs. CLIA (9 Distributed Clients)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Communication Round", fontsize=12)
plt.ylabel("Global Average Accuracy (%)", fontsize=12)
plt.ylim(92, 100.5)
plt.legend(fontsize=11, loc="lower right", frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("fig1_accuracy_convergence.png", dpi=300)
plt.close()
print("▶ [Saved] fig1_accuracy_convergence.png (정확도 수렴 곡선)")

# ==============================================================================
# [그래프 3] 단독 논문용: 통신량 절감 효과 (fig2_communication.png)
# ==============================================================================
fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 1라운드 통신량
bars = ax_bar.bar(["PIA", "CLIA"], [912.38, 0.88], color=["#1f77b4", "#2ca02c"], width=0.4, edgecolor="black", linewidth=1.2)
ax_bar.set_yscale("log")
ax_bar.set_title("Payload per Round (9 Clients)", fontsize=13, fontweight="bold")
ax_bar.set_ylabel("Uplink Traffic (KB) [Log Scale]", fontsize=11)
ax_bar.set_ylim(0.1, 4000)
for bar, val in zip(bars, [912.38, 0.88]):
    ax_bar.text(bar.get_x() + bar.get_width()/2, val * 1.35, f"{val:.2f} KB\n({val*1024:,.0f} B)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_bar.text(0.5, 0.75, "-99.90% Reduction", transform=ax_bar.transAxes, ha="center", fontsize=11, fontweight="bold", color="#d62728",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebeb", edgecolor="#d62728", linewidth=1.2))

# 누적 통신량
ax_cum.plot(pia_rounds, cum_pia_mb, marker="o", color="#1f77b4", linewidth=2.5, label="PIA (Cumulative MB)")
ax_cum.plot(rounds, cum_clia_mb, marker="s", color="#2ca02c", linewidth=2.5, linestyle="--", label="CLIA (Cumulative MB)")
ax_cum.set_title("Cumulative Traffic over 19 Rounds", fontsize=13, fontweight="bold")
ax_cum.set_xlabel("Communication Round", fontsize=11)
ax_cum.set_ylabel("Total Traffic (MB)", fontsize=11)
ax_cum.legend(fontsize=11, loc="upper left", frameon=True)
ax_cum.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("fig2_communication_comparison.png", dpi=300)
plt.close()
print("▶ [Saved] fig2_communication_comparison.png (통신량 비교)")

print("\n모든 그래프 생성이 완료되었습니다!")
