from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import clean_shipments, build_features, FEATURES  # noqa: E402


def main():
    data_dir = ROOT / "data"
    outputs_dir = ROOT / "outputs"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config_path = outputs_dir / "part_c_final_config.csv"
    modelling_path = data_dir / "modelling_table.csv"
    raw_path = data_dir / "raw_shipments.csv"
    hubs_path = data_dir / "hubs.csv"

    for required_path in [config_path, modelling_path, raw_path, hubs_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required project file not found: {required_path}")

    config_df = pd.read_csv(config_path)
    if config_df.empty:
        raise ValueError("part_c_final_config.csv has no configuration row.")
    config = config_df.iloc[0]

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth_raw = str(config["max_depth"]).strip().lower()
    max_depth = None if max_depth_raw in {"unlimited", "none", "nan", ""} else int(max_depth_raw)
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)
    modelling = pd.read_csv(modelling_path)

    if "shipment_id" not in modelling.columns or "is_delayed" not in modelling.columns:
        raise ValueError("modelling_table.csv must contain shipment_id and is_delayed columns.")

    clean, medians = clean_shipments(raw)
    featured = build_features(clean, hubs)

    # raw_shipments.csv has no target column; join the Part A target by shipment_id.
    targets = modelling[["shipment_id", "is_delayed"]].drop_duplicates(subset=["shipment_id"])
    featured = featured.merge(targets, on="shipment_id", how="inner", validate="one_to_one")
    featured["is_delayed"] = pd.to_numeric(featured["is_delayed"], errors="coerce")
    featured = featured.loc[featured["is_delayed"].isin([0, 1])].copy()

    missing_features = [col for col in FEATURES if col not in featured.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    X = featured[FEATURES].copy()
    y = featured["is_delayed"].astype(int)

    # Reproduce Part C's two-stage stratified split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, models_dir / "delay_model.pkl")
    pd.DataFrame({"threshold": [threshold]}).to_csv(models_dir / "delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(models_dir / "feature_order.csv", index=False)
    pd.DataFrame([
        {"column": col, "median": float(medians[col])}
        for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    ]).to_csv(models_dir / "impute_values.csv", index=False)

    reloaded = joblib.load(models_dir / "delay_model.pkl")
    in_memory_probs = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded.predict_proba(X_test)[:, 1]
    max_diff = float(np.max(np.abs(in_memory_probs - reloaded_probs))) if len(X_test) else 0.0

    pd.DataFrame([
        {"metric": "test_rows_scored", "value": int(len(X_test))},
        {"metric": "reload_max_abs_probability_diff", "value": max_diff},
        {"metric": "threshold", "value": threshold},
    ]).to_csv(outputs_dir / "part_d_reload_check.csv", index=False)

    print("Model packaging complete.")
    print(f"Train rows: {len(X_train)} | Validation rows: {len(X_val)} | Test rows: {len(X_test)}")
    print(f"Selected features: {selector_k}")
    print(f"Threshold: {threshold}")
    print(f"Reload max absolute probability difference: {max_diff:.12g}")
    print("Created model files:")
    for name in ["delay_model.pkl", "delay_threshold.csv", "feature_order.csv", "impute_values.csv"]:
        print(f" - {models_dir / name}")
    print(f"Created: {outputs_dir / 'part_d_reload_check.csv'}")

    if max_diff > 1e-9:
        raise RuntimeError("Reloaded model probabilities differ by more than 1e-9.")


if __name__ == "__main__":
    main()
