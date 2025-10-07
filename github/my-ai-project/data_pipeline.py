#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ----------------------------
# Config & Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ----------------------------
# Data Generation (optional)
# ----------------------------
def generate_synthetic_sensor_data(n: int) -> pd.DataFrame:
    """
    Simulate simple sensor streams (temperature, vibration, voltage).
    """
    rng = np.random.default_rng(seed=42)
    ts = pd.date_range(start="2025-01-01", periods=n, freq="H")

    temperature = rng.normal(75, 5, n) + np.linspace(0, 2, n)  # slight drift
    vibration = rng.normal(0.5, 0.1, n) + (temperature - temperature.mean()) * 0.002
    voltage = rng.normal(3.3, 0.05, n) - (temperature - 75) * 0.001

    df = pd.DataFrame({
        "timestamp": ts,
        "temperature": temperature,
        "vibration": vibration,
        "voltage": voltage
    })
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved CSV -> {path}")


# ----------------------------
# ETL Steps
# ----------------------------
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logging.info(f"Loaded raw -> {path} ({len(df)} rows)")
    return df


def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
    # Basic sanity filters (tweak as needed)
    df = df.dropna().copy()
    df = df[df["temperature"].between(50, 100)]
    df = df[df["vibration"].between(0.1, 1.0)]
    df = df[df["voltage"].between(3.0, 3.6)]

    # Feature engineering
    df = df.sort_values("timestamp")
    df["temp_rollmean_24h"] = df["temperature"].rolling(24, min_periods=1).mean()
    df["vibe_rollstd_24h"] = df["vibration"].rolling(24, min_periods=1).std()

    logging.info(f"Cleaned/Transformed rows -> {len(df)}")
    return df


# ----------------------------
# Analysis
# ----------------------------
def run_regression(df: pd.DataFrame) -> dict:
    """
    Simple demo: predict vibration from temperature and voltage.
    """
    valid = df.dropna(subset=["vibration", "temperature", "voltage"])
    X = valid[["temperature", "voltage"]].values
    y = valid["vibration"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    coefs = dict(zip(["temperature", "voltage"], model.coef_))
    intercept = model.intercept_

    logging.info(f"Regression R^2: {r2:.4f} | Coefs: {coefs} | Intercept: {intercept:.4f}")
    return {"r2": r2, "coefs": coefs, "intercept": float(intercept)}


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["temperature", "vibration", "voltage", "temp_rollmean_24h", "vibe_rollstd_24h"]
    cor = df[cols].corr()
    logging.info("Correlation matrix computed.")
    return cor


# ----------------------------
# Visualization
# ----------------------------
def plot_timeseries(df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "sensor_timeseries.png"

    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["temperature"], label="Temperature")
    plt.plot(df["timestamp"], df["vibration"], label="Vibration")
    plt.plot(df["timestamp"], df["voltage"], label="Voltage")
    plt.legend()
    plt.title("Sensor Streams Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logging.info(f"Saved plot -> {outpath}")
    return outpath


def plot_correlation_heatmap(cor: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "correlation_heatmap.png"

    plt.figure(figsize=(6, 5))
    plt.imshow(cor, interpolation="nearest")
    plt.xticks(range(len(cor.columns)), cor.columns, rotation=45, ha="right")
    plt.yticks(range(len(cor.index)), cor.index)
    for i in range(len(cor.index)):
        for j in range(len(cor.columns)):
            plt.text(j, i, f"{cor.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("Correlation Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logging.info(f"Saved plot -> {outpath}")
    return outpath

def plot_regression_fit(df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "regression_fit.png"
    X = df[["temperature"]].values
    y = df["vibration"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(6,4))
    plt.scatter(X, y, alpha=0.5, label="Actual")
    plt.plot(X, y_pred, label="Fitted", linewidth=2)
    plt.title("Temperature vs Vibration – Regression Fit")
    plt.xlabel("Temperature"); plt.ylabel("Vibration"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()
    logging.info(f"Saved plot -> {outpath}")
    return outpath


# ----------------------------
# Orchestration
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-Powered Data Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic raw data")
    parser.add_argument("--rows", type=int, default=1000, help="Rows to generate")
    parser.add_argument("--raw", type=Path, default=Path("data/sensors_raw.csv"))
    parser.add_argument("--clean", type=Path, default=Path("data/sensors_clean.csv"))
    parser.add_argument("--plots", type=Path, default=Path("plots"))
    args = parser.parse_args()

    # Step 0: Generate synthetic data (optional)
    if args.generate:
        raw_df = generate_synthetic_sensor_data(args.rows)
        save_csv(raw_df, args.raw)

    # Step 1: Load raw
    if not args.raw.exists():
        raise FileNotFoundError(f"Raw data not found: {args.raw}. Use --generate to create it.")
    raw_df = load_raw(args.raw)

    # Step 2: Clean & transform
    clean_df = clean_transform(raw_df)
    save_csv(clean_df, args.clean)

    # Step 3: Analysis
    metrics = run_regression(clean_df)
    cor = correlation_matrix(clean_df)

    # Step 4: Plots
    plot_timeseries(clean_df, args.plots)
    plot_correlation_heatmap(cor, args.plots)
    plot_regression_fit(clean_df, args.plots)

    # Step 5: Console summary (copy/paste to README if you want)
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Rows (raw -> clean): {len(raw_df)} -> {len(clean_df)}")
    print(f"R^2: {metrics['r2']:.4f}")
    print(f"Coefs: {metrics['coefs']}")
    print(f"Intercept: {metrics['intercept']:.4f}")
    print("Correlation matrix:\n", cor)
    


if __name__ == "__main__":
    main()
