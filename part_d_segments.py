
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
outputs_dir = ROOT / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

# Load the cleaned modelling table
df = pd.read_csv(ROOT / "data" / "modelling_table.csv")

# 1. Build hub profile, sorted by hub_id
profile = (
    df.groupby("hub_id", as_index=False)
      .agg(
          shipments=("shipment_id", "count"),
          mean_delivery_hours=("delivery_hours", "mean"),
          delay_rate=("is_delayed", "mean"),
          mean_distance_km=("distance_km", "mean"),
          mean_n_stops=("n_stops", "mean"),
      )
      .sort_values("hub_id")
      .reset_index(drop=True)
)

profile.to_csv(outputs_dir / "hub_profile.csv", index=False, float_format="%.4f")

# 2. Scale the four profile features in the required order
cluster_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]
X_scaled = StandardScaler().fit_transform(profile[cluster_features])

# 3. Separation evidence for K = 2, 3, 4, 5
separation_rows = []

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # All unordered hub pairs and their Euclidean distances
    distances = pdist(X_scaled, metric="euclidean")
    pair_i, pair_j = np.triu_indices(len(profile), k=1)
    different = labels[pair_i] != labels[pair_j]

    across_mean = distances[different].mean()
    within_mean = distances[~different].mean()
    ratio = across_mean / within_mean

    sizes = sorted(
        np.bincount(labels).tolist(), reverse=True
    )

    separation_rows.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(map(str, sizes)),
    })

pd.DataFrame(separation_rows).to_csv(
    outputs_dir / "part_d_separation.csv",
    index=False,
    float_format="%.4f"
)

# 4. Fit the required K=3 segmentation
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = km3.fit_predict(X_scaled)

profile["cluster"] = cluster_labels

# Name clusters by ascending mean hub delay rate
cluster_delay = profile.groupby("cluster")["delay_rate"].mean()
ordered_clusters = cluster_delay.sort_values().index.tolist()

segment_names = {
    ordered_clusters[0]: "Standard",
    ordered_clusters[1]: "Watch",
    ordered_clusters[2]: "Escalate",
}

profile["segment"] = profile["cluster"].map(segment_names)

# Attach city and tier from hubs.csv
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
hub_segments = profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left",
    validate="one_to_one",
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values("hub_id")

hub_segments.to_csv(
    outputs_dir / "hub_segments.csv", index=False
)

# Segment profiles are means of the hub-level values
segment_profile = (
    profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

segment_order = {"Standard": 0, "Watch": 1, "Escalate": 2}
segment_profile["_order"] = segment_profile["segment"].map(segment_order)
segment_profile = (
    segment_profile.sort_values("_order")
    .drop(columns="_order")
)

segment_profile.to_csv(
    outputs_dir / "segment_profile.csv",
    index=False,
    float_format="%.4f"
)

# Compare each hub's segment with the segment implied by its tier
tier_to_segment = {
    "Tier 1": "Standard",
    "Tier 2": "Watch",
    "Tier 3": "Escalate",
}

hub_segments["tier_expected_segment"] = (
    hub_segments["hub_tier"].map(tier_to_segment)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["tier_expected_segment"]
].copy()

mismatches.to_csv(
    outputs_dir / "tier_vs_segment_mismatch.csv",
    index=False
)

print("HUB PROFILE")
print(profile.drop(columns=["cluster", "segment"]).round(4).to_string(index=False))

print("\nSEPARATION RESULTS")
print(pd.DataFrame(separation_rows).to_string(index=False))

print("\nHUB SEGMENTS")
print(hub_segments.to_string(index=False))

print("\nSEGMENT PROFILE")
print(segment_profile.round(4).to_string(index=False))

print("\nTIER VS SEGMENT MISMATCH")
print(mismatches.to_string(index=False))

print("\nPart D segmentation files saved successfully.")
