🧠 AI-Powered Data Pipeline for Semiconductor Sensor Analytics
Overview

This project implements an end-to-end AI and predictive-maintenance pipeline for multi-sensor process data. It combines statistical process control (SPC) techniques with machine learning (ML) to detect drift, forecast vibration trends, and support real-world IoT and semiconductor analytics.

Built entirely in Python using pandas, NumPy, scikit-learn, and matplotlib, the pipeline is designed to work with both synthetic and real CSV-based sensor data.

🚀 Core Capabilities
🧩 Data Engineering

Automated data ingestion, cleaning, and transformation for multi-sensor inputs (temperature, voltage, vibration).

Adds lag, delta, and rolling features to capture temporal dependencies.

Implements rolling RMS (vibration energy) and EWMA drift tracking, mirroring SPC methods used in semiconductor fabs.

Integrates FFT-based spectral features for vibration frequency analysis and signal health monitoring.

🤖 Predictive Modeling

Time-aware linear regression modeling with chronological train/test splits to prevent data leakage.

Cross-validation (TimeSeriesSplit) to ensure reliable model generalization.

Multi-metric evaluation: R², MAE, RMSE.

Implements reproducible scikit-learn pipelines (StandardScaler + LinearRegression).

⚠️ Anomaly Detection

Hybrid detection system combining:

Statistical z-score anomalies (|z| > 3)

Unsupervised IsolationForest model for complex multivariate outliers.

Flags potential process drift and early fault conditions.

📊 Visualization Suite

Multi-sensor time-series plots for trend analysis.

3D regression surfaces showing relationships between process variables.

Correlation heatmaps for feature insight.

Anomaly and drift timelines highlighting deviations from nominal behavior.

Residual and error diagnostics for model validation.

⚙️ Methods and Tools
Category	Tools/Techniques
Language	Python
Libraries	pandas, NumPy, scikit-learn, matplotlib
Modeling	Linear Regression, TimeSeriesSplit, IsolationForest
Signal Processing	FFT, Rolling RMS, EWMA
Evaluation Metrics	R², MAE, RMSE
Data Types	Synthetic and real CSV-based sensor data
📂 Output

Cleaned dataset: data/sensors_clean.csv

Generated plots: plots/ directory (time-series, heatmaps, regression surfaces, drift/anomaly timelines, residual diagnostics)

Console model card: prints performance metrics and SPC alerts for quick review.

🎯 Project Purpose

This project demonstrates how an automated ETL → ML → SPC workflow can transform raw sensor streams into interpretable predictive insights.
It bridges the gap between statistical process control and modern AI, providing a scalable foundation for:

Predictive maintenance in semiconductor fabs

Condition monitoring and vibration analytics

Real-time IoT and manufacturing process diagnostics

🔗 Example Visuals

1️⃣ Multi-sensor time-series chart
2️⃣ Correlation heatmap
3️⃣ Regression and residual diagnostics
4️⃣ Anomaly and drift timelines

🧩 Future Work

Add nonlinear models (RandomForest, XGBoost, or Gaussian Process) for enhanced predictive power.

Implement partial dependence plots (PDPs) for explainability.

Integrate model persistence (joblib) and JSON metrics export for reproducibility.

Extend to real hardware telemetry or streaming data for deployment.

👤 Author

Chase R. Mayer
JD, University of Virginia School of Law | MIDS Candidate, UC Berkeley
Exploring the intersection of AI, data systems, and real-world decision-making.