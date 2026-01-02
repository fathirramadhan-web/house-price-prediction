"""EDA: missingness, distributions, correlations, price by district/city.
Writes figures and a text report to outputs/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CATEGORICAL, NUMERIC, TARGET, clean, feature_engineer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "jakarta_house.csv")
    df = feature_engineer(clean(raw))
    report = [
        "EDA — Jakarta House Price Dataset",
        f"Raw shape: {raw.shape} | Clean shape: {df.shape}",
        f"Rows removed: {len(raw) - len(df)}",
        "",
        "Missing values after type coercion (0 room counts -> NaN, before imputation):",
        df[NUMERIC].isna().sum().to_string(),
        "",
        f"Target '{TARGET}' describe (Rupiah):",
        df[TARGET].describe().to_string(),
        "",
        "Numeric feature describe:",
        df[NUMERIC].describe().to_string(),
        "",
    ]

    for col, name in [("district", "top districts"), ("city", "cities")]:
        g = df.groupby(col)[TARGET].agg(["count", "median"]).sort_values("count", ascending=False)
        report += [f"{name} by count / median price:", g.head(20).to_string(), ""]

    report += ["Correlation with price (numeric):",
               df[NUMERIC + [TARGET]].corr()[TARGET].sort_values(ascending=False).to_string()]
    (OUT.parent / "eda_report.txt").write_text("\n".join(report))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(df[TARGET]), bins=60, color="#ef4444", edgecolor="white", linewidth=0.4)
    ax.set_title("Distribusi Harga (log1p Rupiah)"); ax.set_xlabel("log1p(price)"); ax.set_ylabel("Frekuensi")
    fig.tight_layout(); fig.savefig(OUT / "price_distribution.png", dpi=110); plt.close(fig)

    num = df[NUMERIC + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(num, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num))); ax.set_xticklabels(num.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num.columns, fontsize=8)
    for i in range(len(num)):
        for j in range(len(num)):
            v = num.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Korelasi Fitur Numerik")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=110); plt.close(fig)

    g = df.groupby("city")[TARGET].median().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(g.index, g.values / 1e9, color="#dc2626")
    ax.set_title("Median Harga per Kota (miliar Rupiah)"); ax.set_xlabel("Miliar Rupiah")
    fig.tight_layout(); fig.savefig(OUT / "price_by_city.png", dpi=110); plt.close(fig)

    miss = df[NUMERIC].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(miss.index, miss.values, color="#f97316")
    ax.set_title("Missing Value per Fitur (setelah cleaning)"); ax.set_ylabel("Jumlah baris")
    fig.tight_layout(); fig.savefig(OUT / "missing_values.png", dpi=110); plt.close(fig)

    print(f"EDA done: {df.shape[0]} clean rows. Reports -> {OUT.parent / 'eda_report.txt'}, {OUT}")


if __name__ == "__main__":
    main()
