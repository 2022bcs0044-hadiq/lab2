import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "dataset/winequality-red.csv"
MODEL_DIR = "outputs/model"
EVAL_DIR = "outputs/evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv(DATA_PATH, sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Models (3 models)
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel="rbf")
}


results = {}


# -----------------------------
# Train, Evaluate, Save
# -----------------------------
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_regression, k=8)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save trained model
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(pipeline, model_path)

    # Save metrics
    results[name] = {
        "MSE": mse,
        "R2_Score": r2
    }

    # Print metrics to stdout
    print(f"\nModel: {name}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# -----------------------------
# Save Evaluation Results
# -----------------------------
metrics_path = os.path.join(EVAL_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)
