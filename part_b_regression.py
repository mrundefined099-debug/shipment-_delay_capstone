
import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "/content")

from src.shipment_prep import FEATURES

os.makedirs("/content/outputs", exist_ok=True)

# 1. Load data and create fixed train/validation/test splits
df = pd.read_csv("/content/data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# 2. Scaling checks: fit only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

checks = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean",
     float(np.abs(X_train_scaled.mean(axis=0)).max())),
    ("test_scaled_max_abs_mean",
     float(np.abs(X_test_scaled.mean(axis=0)).max())),
]

pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
    "/content/outputs/part_b_checks.csv",
    index=False,
    float_format="%.10f"
)

# Helper to calculate regression metrics
def get_metrics(model, X_data, y_true):
    predictions = model.predict(X_data)
    mse = mean_squared_error(y_true, predictions)

    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, predictions),
    }

# 3. Baseline: predict the training target mean
baseline_prediction = np.full(len(y_val), y_train.mean())
baseline_mse = mean_squared_error(y_val, baseline_prediction)

baseline_val = {
    "MAE": mean_absolute_error(y_val, baseline_prediction),
    "MSE": baseline_mse,
    "RMSE": np.sqrt(baseline_mse),
    "R2": r2_score(y_val, baseline_prediction),
}

# 4. Decision Tree depth experiment
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_results = []
tree_models = {}

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )
    tree.fit(X_train, y_train)

    train_rmse = get_metrics(tree, X_train, y_train)["RMSE"]
    val_rmse = get_metrics(tree, X_val, y_val)["RMSE"]

    depth_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    })

    tree_models[depth] = tree

depth_df = pd.DataFrame(depth_results)
depth_df.to_csv(
    "/content/outputs/part_b_tree_depth.csv",
    index=False,
    float_format="%.4f"
)

# Choose the lowest validation RMSE; ties go to smaller depth
best_depth_row = min(
    depth_results,
    key=lambda row: (
        row["val_rmse"],
        float("inf") if row["max_depth"] == "unlimited"
        else row["max_depth"]
    )
)

best_depth = (
    None if best_depth_row["max_depth"] == "unlimited"
    else int(best_depth_row["max_depth"])
)

best_tree = tree_models[best_depth]

# 5. Train the four candidate models
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "decision_tree": best_tree,
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),
}

metric_rows = [{
    "model": "mean_baseline",
    "split": "validation",
    **baseline_val,
}]

validation_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    scores = get_metrics(model, X_val, y_val)
    validation_scores[name] = scores["RMSE"]

    metric_rows.append({
        "model": name,
        "split": "validation",
        **scores,
    })

# 6. Gradient Boosting learning-rate experiment
gb_rows = []

for rate in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )
    gb.fit(X_train, y_train)
    scores = get_metrics(gb, X_val, y_val)

    gb_rows.append({
        "learning_rate": rate,
        "val_rmse": scores["RMSE"],
        "val_r2": scores["R2"],
    })

pd.DataFrame(gb_rows).to_csv(
    "/content/outputs/part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.4f"
)

# 7. Rank four candidate models by validation RMSE
ranked_models = sorted(
    validation_scores,
    key=lambda name: validation_scores[name]
)

# Cross-validation uses training + validation data only
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for name in ranked_models[:2]:
    cv_model = models[name]

    cv_scores = cross_val_score(
        cv_model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="r2"
    )

    cv_rows.append({
        "model": name,
        "mean_r2": cv_scores.mean(),
        "std_r2": cv_scores.std(),
    })

pd.DataFrame(cv_rows).to_csv(
    "/content/outputs/part_b_cv.csv",
    index=False,
    float_format="%.4f"
)

# 8. Evaluate the validation winner on the test set exactly once
winner = ranked_models[0]
final_test_scores = get_metrics(
    models[winner], X_test, y_test
)

metric_rows.append({
    "model": winner,
    "split": "test",
    **final_test_scores,
})

metrics_df = pd.DataFrame(metric_rows)
metrics_df = metrics_df[
    ["model", "split", "MAE", "MSE", "RMSE", "R2"]
]
metrics_df.to_csv(
    "/content/outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)

# 9. Display results
print("PART B REGRESSION COMPLETED!")
print("\nSplit sizes:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

print("\nScaling checks:")
print(pd.DataFrame(checks, columns=["metric", "value"]).to_string(index=False))

print("\nDecision Tree depth experiment:")
print(depth_df.to_string(index=False))

print("\nModel metrics:")
print(metrics_df.to_string(index=False))

print("\nGradient Boosting learning-rate experiment:")
print(pd.DataFrame(gb_rows).to_string(index=False))

print("\nTop two models: 5-fold cross-validation:")
print(pd.DataFrame(cv_rows).to_string(index=False))

print(f"\nBest model by validation RMSE: {winner}")
print(f"Selected Decision Tree depth: "
      f"{'unlimited' if best_depth is None else best_depth}")
print("\nFiles saved in /content/outputs/")
print(" - part_b_checks.csv")
print(" - part_b_tree_depth.csv")
print(" - part_b_metrics.csv")
print(" - part_b_gb_learning_rate.csv")
print(" - part_b_cv.csv")
