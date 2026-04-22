import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv('../data/cicd_data.csv')
print(f"Loaded {len(df)} builds")
print(f"Failed: {df['failed'].sum()} | Passed: {(df['failed']==0).sum()}")

# ── 2. Features and target ────────────────────────────────
features = [
    'commit_size', 'files_changed', 'test_count',
    'code_coverage', 'hour_of_day', 'day_of_week',
    'prev_build_failed', 'build_duration_mins'
]

X = df[features]
y = df['failed']
print(f"\nTarget distribution:\n{y.value_counts()}")

# ── 3. Train / test split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# ── 4. Scale features ─────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\nFeatures scaled")

# ── 5. Train Random Forest ────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("Model trained!")

# ── 6. Evaluate ───────────────────────────────────────────
y_pred   = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.1f}%")
print("\nDetailed Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Passed', 'Failed']
))

# ── 7. Feature importance ─────────────────────────────────
importance = pd.DataFrame({
    'feature':    features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(importance.to_string(index=False))

# ── 8. Save model ─────────────────────────────────────────
os.makedirs('../saved_model', exist_ok=True)
joblib.dump(model,  '../saved_model/random_forest.pkl')
joblib.dump(scaler, '../saved_model/scaler.pkl')
print("\nModel saved!")