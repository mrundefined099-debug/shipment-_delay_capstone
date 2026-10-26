
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.shipment_prep import clean_shipments, build_features

data_dir = ROOT / "data"
outputs_dir = ROOT / "outputs"
data_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(data_dir / "raw_shipments.csv")
hubs = pd.read_csv(data_dir / "hubs.csv")

# Clean data and build features
clean_df, medians = clean_shipments(raw)
df = build_features(clean_df, hubs)

# Create delay target
df["is_delayed"] = (
    df["delivery_hours"] > df["promised_hours"]
).astype(int)

# Save modelling table
df.to_csv(data_dir / "modelling_table.csv", index=False)

# Basic checks
checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", int(raw.duplicated().sum())),
    ("rows_after_dedup", len(df)),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("delayed_count", int(df["is_delayed"].sum())),
    ("delay_rate", df["is_delayed"].mean()),
]

pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
    outputs_dir / "part_a_checks.csv", index=False
)

# Delay rate by service type, hub tier, and distance band
df["distance_band"] = pd.cut(
    df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

group_results = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = df.groupby(dimension, observed=False).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()

    grouped = grouped.rename(columns={dimension: "group"})
    grouped.insert(0, "dimension", dimension)
    group_results.append(grouped)

pd.concat(group_results, ignore_index=True).to_csv(
    outputs_dir / "part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f",
)

# Numeric correlation matrix
numeric_df = df.select_dtypes(include="number")
numeric_df.corr().to_csv(
    outputs_dir / "part_a_correlations.csv",
    index=True,
    float_format="%.4f",
)

print("Part A completed!")
print("Modelling table:", df.shape)
print("Delayed shipments:", int(df["is_delayed"].sum()))
print("\nFiles created:")
print(data_dir / "modelling_table.csv")
print(outputs_dir / "part_a_checks.csv")
print(outputs_dir / "part_a_delay_by_group.csv")
print(outputs_dir / "part_a_correlations.csv")
