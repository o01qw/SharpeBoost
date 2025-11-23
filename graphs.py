import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR   = r"C:\Users\lazys\Desktop\sharpe_boost"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

csv_path = os.path.join(OUTPUT_DIR, "sharpeboost_results.csv")
df = pd.read_csv(csv_path)

# Optional: if dates were saved as strings, they're fine as labels.
# Create a compact split label for x-axis
df["split_label"] = (
    df["universe"] + " | " +
    df["val_start"].astype(str) + "–" + df["val_end"].astype(str)
)

plt.figure(figsize=(12, 6))

x = range(len(df))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    df["Sharpe_SB_ema_5bps"],
    width=width,
    label="SharpeBoost (EMA, 5bps)"
)
plt.bar(
    [i + width/2 for i in x],
    df["Sharpe_RMSE_ema_5bps"],
    width=width,
    label="RMSE XGBoost (EMA, 5bps)"
)

plt.xticks(x, df["split_label"], rotation=45, ha="right")
plt.ylabel("Validation Sharpe (EMA, 5bps)")
plt.title("SharpeBoost vs RMSE XGBoost by universe/split")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "sharpe_comparison_bars.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))

x = range(len(df))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    df["Turnover_SB"],
    width=width,
    label="SharpeBoost turnover"
)
plt.bar(
    [i + width/2 for i in x],
    df["Turnover_RMSE"],
    width=width,
    label="RMSE XGBoost turnover"
)

plt.xticks(x, df["split_label"], rotation=45, ha="right")
plt.ylabel("Mean turnover")
plt.title("Turnover: SharpeBoost vs RMSE XGBoost")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "turnover_comparison_bars.png"), dpi=200)
plt.close()
print(f"Saved plots to: {OUTPUT_DIR}")