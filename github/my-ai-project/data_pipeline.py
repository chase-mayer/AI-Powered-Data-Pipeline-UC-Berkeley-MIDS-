#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
    Time-aware evaluation:
      - 80/20 chronological split (no shuffling)
      - Standardize inputs
      - Report R²/MAE/RMSE on the held-out test set
      - Optional TimeSeriesSplit CV for robustness
    """
    valid = df.dropna(subset=["vibration", "temperature", "voltage"]).copy()
    X = valid[["temperature", "voltage"]].values
    y = valid["vibration"].values

    # Chronological split (last 20% for test)
    cut = int(0.8 * len(valid))
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    # Test-set metrics
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # Optional: time-series CV (uses full series; still chronological folds)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(pipe, X, y, cv=tscv, scoring="r2")
    metrics = {
        "r2_test": float(r2),
        "mae_test": float(mae),
        "rmse_test": float(rmse),
        "r2_cv_mean": float(cv_scores.mean()),
        "r2_cv_std": float(cv_scores.std())
    }

    # Coefficients (from final fit on train set)
    coefs = dict(zip(["temperature", "voltage"], pipe.named_steps["linreg"].coef_))
    intercept = float(pipe.named_steps["linreg"].intercept_)

    logging.info(
        f"Eval -> R2_test={metrics['r2_test']:.4f} | "
        f"MAE={metrics['mae_test']:.4f} | RMSE={metrics['rmse_test']:.4f} | "
        f"CV_R2={metrics['r2_cv_mean']:.4f}±{metrics['r2_cv_std']:.4f}"
    )
    return {"metrics": metrics, "coefs": coefs, "intercept": intercept}

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["temperature", "vibration", "voltage", "temp_rollmean_24h", "vibe_rollstd_24h"]
    cor = df[cols].corr()
    logging.info("Correlation matrix computed.")
    return cor

def add_semiconductor_features(df: pd.DataFrame,
                               fft_window: int = 128,
                               sampling_rate_hz: float = 1.0) -> pd.DataFrame:
    """
    Add PdM-credible features for fab sensor analytics.
    - Lag & delta features for temporal effects
    - Rolling RMS (condition indicator)
    - EWMA smoothing (SPC-like)
    - Simple spectral features from vibration via rolling FFT
      Note: sampling_rate_hz=1.0 assumes evenly-spaced data (1 unit per step).
    """
    out = df.sort_values("timestamp").copy()

    # ---- Lag & delta
    for col in ["temperature", "vibration", "voltage"]:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_delta"] = out[col].diff()

    # ---- Rolling RMS of vibration (condition indicator)
    out["vibe_rms_24"] = (
        out["vibration"].rolling(24, min_periods=8)
        .apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True)
    )

    # ---- EWMA smoothing (SPC style)
    out["vibration_ewma"] = out["vibration"].ewm(alpha=0.1, adjust=False).mean()
    out["voltage_ewma"]   = out["voltage"].ewm(alpha=0.1, adjust=False).mean()
    out["temperature_ewma"] = out["temperature"].ewm(alpha=0.1, adjust=False).mean()

    # ---- Simple spectral features on rolling windows of vibration
    # We compute dominant frequency index and band energy ratio
    dom_freq_idx = np.full(len(out), np.nan, dtype=float)
    band_ratio   = np.full(len(out), np.nan, dtype=float)

    v = out["vibration"].values.astype(float)
    for i in range(fft_window, len(out)):
        window = v[i-fft_window:i]
        # remove mean to reduce DC component
        window = window - np.mean(window)
        # rFFT magnitude
        fft_mag = np.abs(np.fft.rfft(window))
        # ignore the zero (DC) bin for dominance
        if len(fft_mag) > 1:
            mag_no_dc = fft_mag[1:]
            peak_idx_local = int(np.argmax(mag_no_dc))  # local index ignoring DC
            dom_freq_idx[i] = peak_idx_local + 1  # shift back to global bin index

            # band ratio: low bins vs the rest (heuristic; adjust to your sampling)
            low_band_end = max(2, int(0.1 * len(fft_mag)))  # ~first 10% bins
            low_energy = np.sum(fft_mag[:low_band_end])
            total_energy = np.sum(fft_mag) + 1e-12
            band_ratio[i] = low_energy / total_energy

    out["vibe_domfreq_idx"] = dom_freq_idx
    out["vibe_lowband_ratio"] = band_ratio

    return out

def add_drift_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    EWMA-based drift signal on voltage and vibration.
    Flags points where deviation exceeds rolling (mean + 3*std) of deviation.
    """
    out = df.copy()

    # Deviation from EWMA (absolute)
    out["volt_dev"] = (out["voltage"] - out["voltage_ewma"]).abs()
    out["vibe_dev"] = (out["vibration"] - out["vibration_ewma"]).abs()

    # Rolling baseline of deviation (use ~1 week if hourly; here 168)
    roll = 168
    volt_mu = out["volt_dev"].rolling(roll, min_periods=24).mean()
    volt_sd = out["volt_dev"].rolling(roll, min_periods=24).std()
    vibe_mu = out["vibe_dev"].rolling(roll, min_periods=24).mean()
    vibe_sd = out["vibe_dev"].rolling(roll, min_periods=24).std()

    out["volt_drift_alert"] = (out["volt_dev"] > (volt_mu + 3 * volt_sd)).astype(int)
    out["vibe_drift_alert"] = (out["vibe_dev"] > (vibe_mu + 3 * vibe_sd)).astype(int)

    return out


def add_anomaly_flags(df: pd.DataFrame, use_iforest: bool = True) -> pd.DataFrame:
    """
    Combine simple z-score anomalies on vibration with optional IsolationForest.
    Produces:
      - vibe_z      : standardized vibration
      - vibe_anom_z : 1 if |z|>3
      - vibe_anom_iforest : 1 if IsolationForest flags anomaly (if enabled)
    """
    out = df.copy()

    # Z-score on vibration
    mu = out["vibration"].mean()
    sd = out["vibration"].std(ddof=0) or 1.0
    out["vibe_z"] = (out["vibration"] - mu) / sd
    out["vibe_anom_z"] = (out["vibe_z"].abs() > 3).astype(int)

    if use_iforest and len(out) >= 200:
        # Features for IF: include engineered indicators
        feats = [
            "temperature", "voltage", "vibration",
            "vibe_rms_24", "vibration_ewma",
            "vibe_domfreq_idx", "vibe_lowband_ratio",
            "temperature_ewma", "voltage_ewma"
        ]
        X = out[feats].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").values

        if_model = IsolationForest(
            n_estimators=200, contamination="auto", random_state=42
        )
        scores = if_model.fit_predict(X)  # -1 anomaly, 1 normal
        out["vibe_anom_iforest"] = (scores == -1).astype(int)
    else:
        out["vibe_anom_iforest"] = 0

    return out


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
    order = np.argsort(X.flatten())
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, alpha=0.5, label="Actual")
    plt.plot(X.flatten()[order], y_pred[order], linewidth=2, label="Fitted")
    plt.title("Temperature vs Vibration – Regression Fit")
    plt.xlabel("Temperature"); plt.ylabel("Vibration"); plt.legend(); plt.tight_layout()
    plt.title("Temperature vs Vibration – Regression Fit")
    plt.xlabel("Temperature"); plt.ylabel("Vibration"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()
    logging.info(f"Saved plot -> {outpath}")
    return outpath

def plot_voltage_regression_fit(df: pd.DataFrame, outdir: Path) -> Path:
    """
    Plot Voltage vs. Vibration regression (univariate fit).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "voltage_regression_fit.png"

    X = df[["voltage"]].values
    y = df["vibration"].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, alpha=0.5, label="Actual")
    order = np.argsort(X.flatten())
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, alpha=0.5, label="Actual")
    plt.plot(X.flatten()[order], y_pred[order], linewidth=2, label="Fitted")
    plt.title("Voltage vs Vibration – Regression Fit")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Vibration")
    plt.legend()
    plt.tight_layout()
    plt.title("Voltage vs Vibration – Regression Fit")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Vibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    logging.info(f"Saved plot -> {outpath}")
    return outpath

def plot_regression_surface(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "regression_surface.png"

    X = df[["temperature", "voltage"]].values
    y = df["vibration"].values
    model = LinearRegression().fit(X, y)

    # create meshgrid
    t_range = np.linspace(df["temperature"].min(), df["temperature"].max(), 30)
    v_range = np.linspace(df["voltage"].min(), df["voltage"].max(), 30)
    T, V = np.meshgrid(t_range, v_range)
    Z = model.predict(np.c_[T.ravel(), V.ravel()]).reshape(T.shape)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["temperature"], df["voltage"], df["vibration"], alpha=0.5)
    ax.plot_surface(T, V, Z, color='orange', alpha=0.4)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Voltage (V)")
    ax.set_zlabel("Vibration")
    ax.set_title("Regression Surface: Vibration ~ Temperature + Voltage")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def plot_anomaly_timeline(df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "anomaly_timeline.png"

    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["vibration"], label="Vibration")
    # overlay anomalies (any flag)
    anom = (df["vibe_anom_z"].astype(int) | df["vibe_anom_iforest"].astype(int)) == 1
    plt.scatter(df.loc[anom, "timestamp"], df.loc[anom, "vibration"], marker="x", s=40, label="Anomaly", zorder=3)
    plt.title("Vibration with Anomaly Flags")
    plt.xlabel("Time"); plt.ylabel("Vibration"); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
    return outpath


def plot_drift_timeline(df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "drift_timeline.png"

    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["voltage"], label="Voltage", alpha=0.7)
    plt.plot(df["timestamp"], df["voltage_ewma"], label="Voltage EWMA", linewidth=2)
    drift_idx = df["volt_drift_alert"] == 1
    plt.scatter(df.loc[drift_idx, "timestamp"], df.loc[drift_idx, "voltage"], marker="o", s=30, label="Voltage Drift Alert")

    plt.title("Voltage with EWMA and Drift Alerts")
    plt.xlabel("Time"); plt.ylabel("Voltage (V)"); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
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
    
    # Add PdM / semiconductor features
    clean_df = add_semiconductor_features(clean_df)

    # Drift & anomaly flags
    clean_df = add_drift_flags(clean_df)
    clean_df = add_anomaly_flags(clean_df)

    # Save enriched dataset
    save_csv(clean_df, args.clean)

    # Step 3: Analysis
    metrics = run_regression(clean_df)
    cor = correlation_matrix(clean_df)

    # Step 4: Plots
    plot_timeseries(clean_df, args.plots)
    plot_correlation_heatmap(cor, args.plots)
    plot_regression_fit(clean_df, args.plots)
    plot_voltage_regression_fit(clean_df, args.plots)
    plot_regression_surface(clean_df, args.plots)
    plot_anomaly_timeline(clean_df, args.plots)
    plot_drift_timeline(clean_df, args.plots)


    # Step 5: Console summary (copy/paste to README if you want)
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Rows (raw -> clean): {len(raw_df)} -> {len(clean_df)}")
    print(f"R^2 (test): {metrics['metrics']['r2_test']:.4f} | "
      f"MAE: {metrics['metrics']['mae_test']:.4f} | "
      f"RMSE: {metrics['metrics']['rmse_test']:.4f}")
    print(f"R^2 (CV mean±std): {metrics['metrics']['r2_cv_mean']:.4f}±{metrics['metrics']['r2_cv_std']:.4f}")
    print(f"Coefs: {metrics['coefs']}")
    print(f"Intercept: {metrics['intercept']:.4f}")
    print("Correlation matrix:\n", cor)
    
    print("\n--- OPS/QUALITY SIGNALS ---")
    print(f"Anomalies (z-score): {int(clean_df['vibe_anom_z'].sum())} | (IsolationForest): {int(clean_df['vibe_anom_iforest'].sum())}")
    print(f"Drift alerts (voltage): {int(clean_df['volt_drift_alert'].sum())} | (vibration): {int(clean_df['vibe_drift_alert'].sum())}")
    print("Added features: lag1, delta, RMS(24), EWMA, FFT domfreq idx, low-band ratio")



if __name__ == "__main__":
    main()
