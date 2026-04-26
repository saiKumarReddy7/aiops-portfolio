from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np

app = Flask(__name__)

# Load model at startup
model      = joblib.load('../saved_model/kmeans.pkl')
vectorizer = joblib.load('../saved_model/vectorizer.pkl')
print("Model ready!")

# ── Endpoint 1: Health check ──────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model":  "kmeans_alert_clustering"
    })

# ── Endpoint 2: Deduplicate a batch of alerts ─────────────
@app.route('/deduplicate', methods=['POST'])
def deduplicate():
    data   = request.get_json()
    alerts = data['alerts']

    # Extract just the messages for vectorization
    messages = [a['alert_message'] for a in alerts]

    # Convert to TF-IDF vectors
    X = vectorizer.transform(messages)

    # Predict cluster for each alert
    clusters = model.predict(X)

    # Group alerts by cluster
    cluster_groups = {}
    for i, cluster_id in enumerate(clusters):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append({
            'index':   i,
            'alert':   alerts[i],
            'vector':  X[i]
        })

    # Pick best representative from each cluster
    deduplicated = []
    for cluster_id, group in cluster_groups.items():
        center      = model.cluster_centers_[cluster_id]
        vectors     = [item['vector'] for item in group]

        from scipy.sparse import vstack
        cluster_matrix = vstack(vectors)
        similarities   = cosine_similarity(
            cluster_matrix, center.reshape(1, -1)
        )
        best_idx   = similarities.argmax()
        best_alert = group[best_idx]['alert']

        deduplicated.append({
            "cluster_id":           cluster_id,
            "representative_alert": best_alert,
            "total_in_cluster":     len(group),
            "suppressed_count":     len(group) - 1,
            "all_alerts_in_cluster": [g['alert'] for g in group]
        })

    total_suppressed = sum(d['suppressed_count'] for d in deduplicated)

    return jsonify({
        "original_alert_count":     len(alerts),
        "unique_alert_count":       len(deduplicated),
        "total_suppressed":         total_suppressed,
        "noise_reduction_percent":  round(total_suppressed / len(alerts) * 100),
        "deduplicated_alerts":      deduplicated
    })

# ── Endpoint 3: Classify a single new alert ───────────────
@app.route('/classify', methods=['POST'])
def classify():
    data    = request.get_json()
    message = data['alert_message']

    # Convert to vector and predict cluster
    vector     = vectorizer.transform([message])
    cluster_id = int(model.predict(vector)[0])

    # Get similarity to cluster center
    center      = model.cluster_centers_[cluster_id]
    similarity  = float(cosine_similarity(vector, center.reshape(1, -1))[0][0])

    return jsonify({
        "alert_message": message,
        "cluster_id":    cluster_id,
        "similarity_to_cluster_center": round(similarity, 4),
        "confidence":    "HIGH" if similarity > 0.5 else "MEDIUM" if similarity > 0.2 else "LOW"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)