"""California Housing: linear regression baseline and improvements.

Runs every experiment on the SAME untouched test set so results are comparable:
  1. Baseline (always predict the training mean)
  2. Linear regression
  3. Linear regression trained on outlier-filtered data (tested on the full test set)
  4. Linear regression with clipped features (limits learned from train only)
  5. Random Forest (non-linear reference)

Usage:  python src/train.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def record(name, y_true, y_pred):
    return {"model": name, "RMSE": round(rmse(y_true, y_pred), 3),
            "R2": round(float(r2_score(y_true, y_pred)), 3)}


def residual_plot(y_true, y_pred, title, path):
    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_pred, y_true - y_pred, alpha=0.3)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted (x $100k)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    (ROOT / "images").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    rows = []

    # 1. Baseline
    base_pred = np.full(len(y_test), y_train.mean())
    rows.append(record("Baseline (mean)", y_test, base_pred))

    # 2. Linear regression
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    rows.append(record("Linear regression", y_test, lin_pred))
    print("Train R2:", round(r2_score(y_train, lin.predict(X_train)), 3),
          "| Test R2:", round(r2_score(y_test, lin_pred), 3))
    residual_plot(y_test, lin_pred, "Linear regression residuals",
                  ROOT / "images" / "residuals_linear.png")

    # 3. Train on outlier-filtered rows, evaluate on the full, untouched test set
    keep = (X_train["AveRooms"] < 10) & (X_train["AveOccup"] < 10) & (X_train["AveBedrms"] < 3)
    filt = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train[keep], y_train[keep])
    rows.append(record("Train filtered, test full", y_test, filt.predict(X_test)))

    # 4. Clipping: limits come from the training set only (no leakage)
    lo, hi = X_train.quantile(0.01), X_train.quantile(0.99)
    clip = make_pipeline(StandardScaler(), LinearRegression()).fit(
        X_train.clip(lo, hi, axis=1), y_train)
    clip_pred = clip.predict(X_test.clip(lo, hi, axis=1))
    rows.append(record("Linear + clipping (1%-99%)", y_test, clip_pred))
    residual_plot(y_test, clip_pred, "Linear + clipping residuals",
                  ROOT / "images" / "residuals_clipped.png")

    # 5. Non-linear reference
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append(record("Random Forest", y_test, rf.predict(X_test)))

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "results" / "metrics.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
