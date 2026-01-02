"""Streamlit UI: single prediction, batch CSV, model comparison, EDA.

Run: .venv/bin/streamlit run streamlit_app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import data as d

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics.json"

st.set_page_config(page_title="Prediksi Harga Rumah Jakarta", page_icon="🏠", layout="wide")
st.title("Prediksi Harga Rumah Jakarta")
st.caption("Linear Regression · Random Forest · Gradient Boosting — dataset Kaggle Jakarta House Price")


@st.cache_resource
def get_model(name: str):
    path = MODELS / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def get_metrics():
    return json.loads(METRICS.read_text()) if METRICS.exists() else None


def default_winner() -> str:
    m = get_metrics()
    return (m["winner"] + "_tuned") if m else "random_forest_tuned"


def model_options() -> list[str]:
    if not MODELS.exists():
        return []
    return sorted(p.stem for p in MODELS.glob("*.joblib"))


def predict_one(row: dict, model) -> float:
    from inference import predict_prices
    return float(predict_prices(pd.DataFrame([row]), model)[0])


opts = model_options()
if not opts:
    st.error("Model belum ada. Jalankan `.venv/bin/python train.py` dulu.")
    st.stop()

tabs = st.tabs(["Prediksi Tunggal", "Batch CSV", "Evaluasi Model", "EDA & Data"])

with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.selectbox("Model", opts, index=opts.index(default_winner()) if default_winner() in opts else 0)
        district = st.text_input("Kecamatan (district)", "Kelapa Gading")
        city = st.selectbox("Kota (city)", [
            "Jakarta Pusat", "Jakarta Utara", "Jakarta Barat", "Jakarta Selatan", "Jakarta Timur",
            "Bogor", "Depok", "Tangerang", "Bekasi", "Unknown",
        ])
        c1, c2, c3 = st.columns(3)
        bed = c1.number_input("Kamar tidur", 0, 20, 3)
        bath = c2.number_input("Kamar mandi", 0, 20, 2)
        carport = c3.number_input("Carport", 0, 20, 1)
        c4, c5 = st.columns(2)
        land = c4.number_input("Luas tanah (m²)", 1, 100000, 120)
        bldg = c5.number_input("Luas bangunan (m²)", 1, 100000, 150)
    with col2:
        st.markdown("### Estimasi")
        model = get_model(name)
        if st.button("Prediksi", use_container_width=True) and model is not None:
            try:
                price = predict_one({
                    "district": district, "city": city, "bed_rooms": bed, "bath_rooms": bath,
                    "carport": carport, "land_area": land, "building_area": bldg,
                }, model)
                st.metric("Harga estimasi", f"Rp {price:,.0f}")
                st.caption(f"≈ Rp {price / 1e9:.2f} miliar · model: {name}")
            except Exception as e:
                st.error(f"Gagal prediksi: {e}")
        else:
            st.info("Isi form lalu klik Prediksi.")

with tabs[1]:
    st.write("Upload CSV dengan kolom: district, city, bed_rooms, bath_rooms, carport, land_area, building_area")
    up = st.file_uploader("CSV input", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        st.write("Preview:", raw.head())
        name_b = st.selectbox("Model (batch)", opts, key="batch_model",
                              index=opts.index(default_winner()) if default_winner() in opts else 0)
        model_b = get_model(name_b)
        if st.button("Jalankan batch prediction", use_container_width=True) and model_b is not None:
            try:
                from predict import predict_frame
                out = predict_frame(raw, model_b)
                st.success(f"{len(out)} baris berhasil diprediksi")
                st.dataframe(out[["predicted_price"] + [c for c in d.NUMERIC if c in out]].head(20))
                st.download_button("Download hasil", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Gagal: {e}")

with tabs[2]:
    m = get_metrics()
    if not m:
        st.info("Belum ada metrics.json. Jalankan train.py.")
    else:
        st.metric("Model terbaik", m["winner"], help=m.get("note", ""))
        st.caption(m.get("note", ""))
        rows = []
        for variant in ("baseline", "tuned"):
            for name_, s in m.get(variant, {}).items():
                rows.append({"model": name_, "variant": variant, "RMSE (Rp)": round(s["rmse"]),
                             "MAE (Rp)": round(s["mae"]), "R²": round(s["r2"], 4)})
        dfm = pd.DataFrame(rows)
        st.dataframe(dfm, hide_index=True)
        plot_df = dfm[dfm["variant"] == "tuned"].set_index("model")[["RMSE (Rp)", "R²"]]
        st.bar_chart(plot_df["RMSE (Rp)"])
        with st.expander("Hyperparameter terbaik"):
            st.json(m.get("best_params", {}))

with tabs[3]:
    m = get_metrics()
    if m:
        st.json({"n_rows": m["n_rows"], "train": m["train_rows"], "test": m["test_rows"],
                 "features": m.get("features")})
    figdir = ROOT / "outputs" / "figures"
    if figdir.exists():
        for f in sorted(figdir.glob("*.png")):
            st.subheader(f.stem.replace("_", " ").title())
            st.image(str(f))
        rpt = figdir.parent / "eda_report.txt"
        if rpt.exists():
            with st.expander("Laporan EDA (teks)"):
                st.text(rpt.read_text()[:12000])
