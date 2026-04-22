from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model at startup
model  = joblib.load('../saved_model/random_forest.pkl')
scaler = joblib.load('../saved_model/scaler.pkl')
print("Model ready!")

# ── Endpoint 1: Health check ──────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model":  "random_forest_cicd"
    })

# ── Endpoint 2: Single prediction ─────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [[
        data['commit_size'],
        data['files_changed'],
        data['test_count'],
        data['code_coverage'],
        data['hour_of_day'],
        data['day_of_week'],
        data['prev_build_failed'],
        data['build_duration_mins']
    ]]

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0]

    failure_prob = round(float(probability[1]) * 100, 1)
    pass_prob    = round(float(probability[0]) * 100, 1)

    return jsonify({
        "af":"bf",
        "prediction":          "WILL FAIL" if prediction == 1 else "WILL PASS",
        "failure_probability": f"{failure_prob}%",
        "pass_probability":    f"{pass_prob}%",
        "recommendation":      get_recommendation(failure_prob),
        "input":               data
    })

# ── Endpoint 3: Batch prediction ──────────────────────────
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    builds  = request.get_json()['builds']
    results = []

    for build in builds:
        features = [[
            build['commit_size'],
            build['files_changed'],
            build['test_count'],
            build['code_coverage'],
            build['hour_of_day'],
            build['day_of_week'],
            build['prev_build_failed'],
            build['build_duration_mins']
        ]]

        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probability     = model.predict_proba(features_scaled)[0]
        failure_prob    = round(float(probability[1]) * 100, 1)

        results.append({
            "build_id":            build.get('build_id', 'unknown'),
            "prediction":          "WILL FAIL" if prediction == 1 else "WILL PASS",
            "failure_probability": f"{failure_prob}%",
            "recommendation":      get_recommendation(failure_prob)
        })

    high_risk = sum(1 for r in results if "FAIL" in r['prediction'])

    return jsonify({
        "total_builds":    len(results),
        "high_risk_count": high_risk,
        "safe_count":      len(results) - high_risk,
        "results":         results
    })

# ── Helper function ────────────────────────────────────────
def get_recommendation(failure_prob):
    if failure_prob >= 70:
        return "HIGH RISK — review large commit, increase coverage"
    elif failure_prob >= 40:
        return "MEDIUM RISK — consider splitting into smaller PRs"
    else:
        return "LOW RISK — safe to run pipeline"

if __name__ == '__main__':
    app.run(debug=True, port=5000)