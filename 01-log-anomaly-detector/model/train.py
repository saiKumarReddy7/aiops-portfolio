import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── 1. Load data ──────────────────────────────────────
df = pd.read_csv('../data/server_logs.csv')
print(f"Loaded {len(df)} log entries")
print(df.head(3))

# ── 2. Select features ────────────────────────────────
features = ['response_time_ms', 'status_code',
            'cpu_usage', 'memory_usage', 'error_count']
X = df[features]
print("\nFeatures selected:")
print(X.head(3))

# ── 3. Scale the features ─────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nAfter scaling (first 3 rows):")
print(X_scaled[:3])

# ── 4. Train the model ────────────────────────────────
model = IsolationForest(
    contamination=0.1,
    random_state=42,
    n_estimators=100
)
model.fit(X_scaled)
print("\nModel trained successfully!")

# ── 5. Predict ────────────────────────────────────────
predictions = model.predict(X_scaled)
scores      = model.score_samples(X_scaled)

df['prediction'] = predictions
df['score']      = scores
df['is_anomaly'] = df['prediction'].apply(
    lambda x: 'ANOMALY' if x == -1 else 'normal'
)

print("\nFull results:")
print(df[['timestamp','response_time_ms','error_count','is_anomaly','score']])

anomaly_count = (df['is_anomaly'] == 'ANOMALY').sum()
print(f"\nFound {anomaly_count} anomalies out of {len(df)} log entries")

# ── 6. Save model ─────────────────────────────────────
os.makedirs('../saved_model', exist_ok=True)
joblib.dump(model,  '../saved_model/isolation_forest.pkl')
joblib.dump(scaler, '../saved_model/scaler.pkl')
print("\nModel saved to saved_model/")