from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path("/content")
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import clean_shipments, build_features

def main():
    data = ROOT / "data"
    outputs = ROOT / "outputs"
    models = ROOT / "models"
    outputs.mkdir(exist_ok=True)

    new_shipments = pd.read_csv(data / "new_shipments.csv")
    hubs = pd.read_csv(data / "hubs.csv")
    segments = pd.read_csv(outputs / "hub_segments.csv")

    medians_df = pd.read_csv(models / "impute_values.csv")
    medians = dict(zip(medians_df["column"], medians_df["median"]))
    features = pd.read_csv(models / "feature_order.csv")["feature"].tolist()
    threshold = float(pd.read_csv(models / "delay_threshold.csv")["threshold"].iloc[0])
    model = joblib.load(models / "delay_model.pkl")

    clean, _ = clean_shipments(new_shipments, medians=medians)
    featured = build_features(clean, hubs)

    probabilities = model.predict_proba(featured[features])[:, 1]
    flagged = (probabilities >= threshold).astype(int)

    result = pd.DataFrame({
        "shipment_id": featured["shipment_id"],
        "hub_id": featured["hub_id"],
        "segment": featured["hub_id"].map(
            segments.set_index("hub_id")["segment"]
        ),
        "delay_probability": np.round(probabilities, 4),
        "flagged": flagged,
    })

    result["action"] = np.select(
        [
            (result["flagged"] == 1) & (result["segment"] == "Escalate"),
            result["flagged"] == 1,
        ],
        ["Escalate to hub manager", "Proactive customer call"],
        default="No action",
    )

    result = result[
        ["shipment_id", "hub_id", "segment",
         "delay_probability", "flagged", "action"]
    ]

    if len(result) != 48:
        raise ValueError(f"Expected 48 shipments, got {len(result)}")
    if result.isna().any().any() or result["shipment_id"].duplicated().any():
        raise ValueError("Output contains missing values or duplicate shipment IDs.")

    result.to_csv(outputs / "new_shipment_actions.csv", index=False)
    print("Success! 48 shipment predictions saved.")
    print("Flagged:", int(result["flagged"].sum()))
    print(result["action"].value_counts().to_string())
    print(result.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
