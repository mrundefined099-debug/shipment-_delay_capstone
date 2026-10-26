
from pathlib import Path
import sys
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.shipment_prep import FEATURES, NOISE_COLUMNS

outputs_dir = ROOT / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

# 1. Load data and make stratified train/validation/test splits
df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("PART C DATA SPLIT")
print("Train:", X_train.shape, "| delayed:", int(y_train.sum()))
print("Validation:", X_val.shape, "| delayed:", int(y_val.sum()))
print("Test:", X_test.shape, "| delayed:", int(y_test.sum()))

# Common metrics helper
def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

# 2. Blind baseline and three classifiers
model_rows = []

blind_pred = np.zeros(len(y_val), dtype=int)
model_rows.append({
    "model": "blind_all_on_time",
    **classification_metrics(y_val, blind_pred),
})

logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])
logistic.fit(X_train, y_train)
model_rows.append({
    "model": "logistic_regression",
    **classification_metrics(y_val, logistic.predict(X_val)),
})

# Select tree depth using validation F1; ties choose smaller depth
tree_candidates = []
for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini", max_depth=depth, random_state=42
    )
    tree.fit(X_train, y_train)
    pred = tree.predict(X_val)
    tree_candidates.append({
        "depth": depth,
        "f1": f1_score(y_val, pred, zero_division=0),
        "model": tree,
    })

best_tree_result = max(
    tree_candidates, key=lambda row: (row["f1"], -row["depth"])
)
best_tree = best_tree_result["model"]

model_rows.append({
    "model": f"decision_tree_depth_{best_tree_result['depth']}",
    **classification_metrics(y_val, best_tree.predict(X_val)),
})

forest = RandomForestClassifier(
    n_estimators=100, random_state=42
)
forest.fit(X_train, y_train)
model_rows.append({
    "model": "random_forest_default",
    **classification_metrics(y_val, forest.predict(X_val)),
})

pd.DataFrame(model_rows).to_csv(
    outputs_dir / "part_c_models.csv",
    index=False, float_format="%.4f"
)

# 3. Feature scores and default-forest importances
selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)

feature_score_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": forest.feature_importances_,
})

feature_score_df["f_rank"] = (
    feature_score_df["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)
feature_score_df["is_noise"] = (
    feature_score_df["feature"].isin(NOISE_COLUMNS)
)

feature_score_df = feature_score_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
].sort_values(["f_rank", "feature"])

feature_score_df.to_csv(
    outputs_dir / "part_c_feature_scores.csv",
    index=False, float_format="%.4f"
)

# 4. SelectKBest k comparison
selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    candidate = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42
        )),
    ])
    candidate.fit(X_train, y_train)
    pred = candidate.predict(X_val)
    selectk_rows.append({
        "k": k,
        "val_f1": f1_score(y_val, pred, zero_division=0),
    })

pd.DataFrame(selectk_rows).to_csv(
    outputs_dir / "part_c_selectk.csv",
    index=False, float_format="%.4f"
)

# From this point, use k=7.
# Fit the selector only on training rows.
selector_7 = SelectKBest(f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

# 5. Grid search and randomized search
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
)
grid_search.fit(X_train_7, y_train)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train_7, y_train)

default_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
)
default_7.fit(X_train_7, y_train)

tuning_rows = []

for name, candidates, folds, total_fits, params, cv_score, model in [
    (
        "grid_search", 27, 5, 135,
        grid_search.best_params_, grid_search.best_score_,
        grid_search.best_estimator_
    ),
    (
        "randomized_search", 9, 5, 45,
        random_search.best_params_, random_search.best_score_,
        random_search.best_estimator_
    ),
    (
        "default_forest_same_7_features", 1, 0, 1,
        {
            "n_estimators": 100,
            "max_depth": "unlimited",
            "min_samples_leaf": 1
        },
        np.nan, default_7
    ),
]:
    pred = model.predict(X_val_7)
    tuning_rows.append({
        "search": name,
        "candidates": candidates,
        "cv_folds": folds,
        "total_fits": total_fits,
        "best_params": json.dumps(params, sort_keys=True),
        "best_cv_accuracy": cv_score,
        "val_f1": f1_score(y_val, pred, zero_division=0),
    })

tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(
    outputs_dir / "part_c_tuning.csv",
    index=False, float_format="%.4f"
)

# 6. Choose the highest validation F1.
# Tie priority: default forest, grid search, randomized search.
priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}
winner_row = sorted(
    tuning_rows,
    key=lambda row: (-row["val_f1"], priority[row["search"]])
)[0]

chosen_from = winner_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

# Build the final pipeline and fit on training rows only
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42, **chosen_params
    )),
])
final_model.fit(X_train, y_train)

# 7. Select threshold using validation recall >= 0.80
val_probabilities = final_model.predict_proba(X_val)[:, 1]
threshold_rows = []

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (val_probabilities >= threshold).astype(int)
    threshold_rows.append({
        "threshold": threshold,
        **classification_metrics(y_val, pred),
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    outputs_dir / "part_c_thresholds.csv",
    index=False, float_format="%.4f"
)

eligible = threshold_df[threshold_df["recall"] >= 0.80]
chosen_threshold = (
    float(eligible["threshold"].max()) if len(eligible) else 0.1
)

# 8. Final test evaluation once, at the selected threshold
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_probabilities >= chosen_threshold).astype(int)

test_result = {
    "threshold": chosen_threshold,
    **classification_metrics(y_test, test_pred),
}

pd.DataFrame([test_result]).to_csv(
    outputs_dir / "part_c_final_test.csv",
    index=False, float_format="%.4f"
)

# Save final configuration
config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "unlimited" if chosen_params.get("max_depth") is None
        else chosen_params.get("max_depth")
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])

config.to_csv(
    outputs_dir / "part_c_final_config.csv",
    index=False
)

# 9. Display summary
print("\nCLASSIFICATION MODEL RESULTS")
print(pd.DataFrame(model_rows).to_string(index=False))

print("\nFEATURE SCORES")
print(feature_score_df.to_string(index=False))

print("\nSELECT K RESULTS")
print(pd.DataFrame(selectk_rows).to_string(index=False))

print("\nTUNING RESULTS")
print(tuning_df.to_string(index=False))

print("\nTHRESHOLD RESULTS")
print(threshold_df.to_string(index=False))

print("\nCHOSEN CONFIGURATION")
print(config.to_string(index=False))

print("\nFINAL TEST RESULT")
print(pd.DataFrame([test_result]).to_string(index=False))

print("\nPart C completed. Files saved:")
for filename in [
    "part_c_models.csv",
    "part_c_feature_scores.csv",
    "part_c_selectk.csv",
    "part_c_tuning.csv",
    "part_c_thresholds.csv",
    "part_c_final_test.csv",
    "part_c_final_config.csv",
]:
    print(" -", filename)
