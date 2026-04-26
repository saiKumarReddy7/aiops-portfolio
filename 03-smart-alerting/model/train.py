import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import numpy as np

# ── 1. Load data ───────────────────────────────────────────
df = pd.read_csv('../data/alerts.csv')
print(f"Loaded {len(df)} alerts")
print(f"Services: {df['service'].unique()}")

# ── 2. Convert alert text to numbers (TF-IDF) ─────────────
# TF-IDF = Term Frequency - Inverse Document Frequency
# Converts text into numerical vectors the model can understand
# Common words like "the", "is" get low scores
# Unique important words like "payment", "database" get high scores
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['alert_message'])
print(f"\nText converted to matrix: {X.shape}")

# ── 3. Determine number of clusters ───────────────────────
# We expect 4 unique alert types (payment, database, cache, api)
# In real systems you'd use the Elbow Method to find this
n_clusters = 4
print(f"Number of clusters: {n_clusters}")

# ── 4. Train K-Means ──────────────────────────────────────
# K-Means groups similar alerts into n_clusters groups
# Each group = one unique type of problem
model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
model.fit(X)
print("K-Means model trained!")

# ── 5. Assign clusters to alerts ──────────────────────────
df['cluster'] = model.labels_
print("\nAlerts with cluster assignments:")
print(df[['alert_id', 'service', 'cluster', 'alert_message']])

# ── 6. Show what each cluster contains ────────────────────
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)
for cluster_id in range(n_clusters):
    cluster_alerts = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_alerts)} alerts):")
    for _, row in cluster_alerts.iterrows():
        print(f"  [{row['severity']}] {row['service']}: {row['alert_message'][:50]}...")

# ── 7. Deduplicate — pick one alert per cluster ────────────
# Pick the most central alert in each cluster
# (closest to the cluster center = most representative)
print("\n" + "="*60)
print("DEDUPLICATED ALERTS (what engineers actually see)")
print("="*60)

deduplicated = []
for cluster_id in range(n_clusters):
    cluster_mask   = df['cluster'] == cluster_id
    cluster_alerts = df[cluster_mask]
    cluster_matrix = X[cluster_mask.values]

    # Get the center of this cluster
    center = model.cluster_centers_[cluster_id]

    # Calculate similarity of each alert to the center
    similarities  = cosine_similarity(cluster_matrix, center.reshape(1, -1))
    best_idx      = similarities.argmax()
    best_alert    = cluster_alerts.iloc[best_idx]

    deduplicated.append({
        'cluster_id':    cluster_id,
        'alert_count':   len(cluster_alerts),
        'severity':      best_alert['severity'],
        'service':       best_alert['service'],
        'representative_alert': best_alert['alert_message'],
        'suppressed_count': len(cluster_alerts) - 1
    })
    print(f"\nCluster {cluster_id} — {len(cluster_alerts)} alerts → 1 sent")
    print(f"  Service:  {best_alert['service']}")
    print(f"  Severity: {best_alert['severity']}")
    print(f"  Alert:    {best_alert['alert_message']}")
    print(f"  Suppressed {len(cluster_alerts)-1} duplicate alerts")

dedup_df = pd.DataFrame(deduplicated)
total_suppressed = dedup_df['suppressed_count'].sum()
print(f"\n{'='*60}")
print(f"SUMMARY: {len(df)} alerts → {n_clusters} unique alerts")
print(f"Noise reduction: {total_suppressed} duplicate alerts suppressed")
print(f"Noise reduction %: {round(total_suppressed/len(df)*100)}%")

# ── 8. Save model ─────────────────────────────────────────
os.makedirs('../saved_model', exist_ok=True)
joblib.dump(model,      '../saved_model/kmeans.pkl')
joblib.dump(vectorizer, '../saved_model/vectorizer.pkl')
print("\nModel saved!")