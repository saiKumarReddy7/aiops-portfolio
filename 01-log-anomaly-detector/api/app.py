from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model  = joblib.load('../saved_model/isolation_forest.pkl')
scaler = joblib.load('../saved_model/scaler.pkl')
print("Model loaded and ready!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model":  "isolation_forest"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [[
        data['response_time_ms'],
        data['status_code'],
        data['cpu_usage'],
        data['memory_usage'],
        data['error_count']
    ]]

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    score           = model.score_samples(features_scaled)[0]

    return jsonify({
        "is_anomaly": bool(prediction == -1),
        "label":      "ANOMALY" if prediction == -1 else "normal",
        "score":      round(float(score), 4),
        "input":      data
    })

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    data    = request.get_json()
    logs    = data['logs']
    results = []

    for log in logs:
        features = [[
            log['response_time_ms'],
            log['status_code'],
            log['cpu_usage'],
            log['memory_usage'],
            log['error_count']
        ]]
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        score           = model.score_samples(features_scaled)[0]

        results.append({
            "timestamp":  log.get('timestamp', 'unknown'),
            "is_anomaly": bool(prediction == -1),
            "label":      "ANOMALY" if prediction == -1 else "normal",
            "score":      round(float(score), 4)
        })

    anomaly_count = sum(1 for r in results if r['is_anomaly'])

    return jsonify({
        "total_logs":    len(results),
        "anomaly_count": anomaly_count,
        "normal_count":  len(results) - anomaly_count,
        "results":       results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)