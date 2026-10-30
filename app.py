
"""
Customer Segmentation System - Flask Web Application
===================================================
A modern, responsive Flask web application for customer segmentation.
Features:
- Home: Project overview, customer segmentation concepts, methodology.
- Dashboard: Real-time KPIs, segment distributions, Chart.js visualizations.
- Customer Analysis: Interactive customer table, ID search, cluster filters, segment profiles.
- Prediction: Real-time K-Means segment classification & Logistic Regression spend scoring.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'customer-segmentation-secret-key-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'Mall_Customers_Segmented.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(DATA_DIR, 'Mall_Customers.csv')

# Global variables for loaded models
kmeans_model = None
scaler = None
logistic_model = None
logistic_scaler = None
segment_metadata = None
df_customers = None


def load_application_models():
    """
    Loads saved Joblib models, feature scalers, and segmented customer data once at startup.
    """
    global kmeans_model, scaler, logistic_model, logistic_scaler, segment_metadata, df_customers

    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    log_path = os.path.join(MODELS_DIR, 'logistic_model.joblib')
    log_scaler_path = os.path.join(MODELS_DIR, 'logistic_scaler.joblib')
    meta_path = os.path.join(MODELS_DIR, 'segment_metadata.joblib')

    try:
        if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
            kmeans_model = joblib.load(kmeans_path)
            scaler = joblib.load(scaler_path)
            print("[INFO] K-Means model and Scaler loaded successfully.")

        if os.path.exists(log_path) and os.path.exists(log_scaler_path):
            logistic_model = joblib.load(log_path)
            logistic_scaler = joblib.load(log_scaler_path)
            print("[INFO] Logistic Regression model and Scaler loaded successfully.")

        if os.path.exists(meta_path):
            segment_metadata = joblib.load(meta_path)
            print("[INFO] Segment metadata loaded successfully.")

        if os.path.exists(DATA_PATH):
            df_customers = pd.read_csv(DATA_PATH)
            print(f"[INFO] Loaded {len(df_customers)} customer records from {DATA_PATH}.")
    except Exception as e:
        print(f"[ERROR] Error loading models or data: {e}")


# Initialize models at launch
load_application_models()


# =====================================================================
# ROUTES
# =====================================================================

@app.route('/')
def index():
    """
    Home page introducing the Customer Segmentation System.
    """
    summary = segment_metadata.get('dataset_summary', {}) if segment_metadata else {}
    return render_template('index.html', summary=summary, metadata=segment_metadata)


@app.route('/dashboard')
def dashboard():
    """
    Dashboard presenting KPIs, summary statistics, cluster breakdowns,
    and interactive visual charts.
    """
    if segment_metadata is None or df_customers is None:
        flash("Models or data not found. Please run 'python train_model.py' first.", "warning")
        return redirect(url_for('index'))

    summary = segment_metadata.get('dataset_summary', {})
    segments = segment_metadata.get('segments', {})
    clf_metrics = segment_metadata.get('classification_metrics', {})

    # Age distribution buckets for dashboard chart
    age_bins = [18, 25, 35, 50, 71]
    age_labels = ['18-24', '25-34', '35-49', '50+']
    df_customers['Age_Group'] = pd.cut(df_customers['Age'], bins=age_bins, labels=age_labels, right=False)
    age_distribution = df_customers['Age_Group'].value_counts()[age_labels].to_dict()

    # Gender breakdown
    gender_counts = df_customers['Gender'].value_counts().to_dict()

    return render_template(
        'dashboard.html',
        summary=summary,
        segments=segments,
        clf_metrics=clf_metrics,
        age_distribution=age_distribution,
        gender_counts=gender_counts
    )


@app.route('/customers')
def customers():
    """
    Customer Analysis view:
    Allows searching by Customer ID, filtering by cluster segment,
    and reviewing individual customer attributes and segment characteristics.
    """
    if df_customers is None or segment_metadata is None:
        flash("Customer data unavailable. Please run train_model.py.", "danger")
        return redirect(url_for('index'))

    # Query parameters for filtering and searching
    search_id = request.args.get('search_id', '').strip()
    selected_cluster = request.args.get('cluster', 'all').strip()

    filtered_df = df_customers.copy()

    # Filter by Cluster
    if selected_cluster != 'all' and selected_cluster != '':
        try:
            c_int = int(selected_cluster)
            filtered_df = filtered_df[filtered_df['Cluster'] == c_int]
        except ValueError:
            pass

    # Search by ID
    if search_id:
        try:
            cid = int(search_id)
            filtered_df = filtered_df[filtered_df['CustomerID'] == cid]
        except ValueError:
            flash("Customer ID must be a valid number.", "info")

    records = filtered_df.to_dict(orient='records')
    segments = segment_metadata.get('segments', {})

    return render_template(
        'customers.html',
        customers=records,
        total_matched=len(records),
        total_records=len(df_customers),
        segments=segments,
        selected_cluster=selected_cluster,
        search_id=search_id
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Customer Segment Prediction form and result display.
    Accepts Annual Income and Spending Score, validates input,
    scales features, and runs K-Means prediction.
    Also accepts Age and Gender to evaluate High-Spending propensity via Logistic Regression.
    """
    prediction_result = None
    form_data = {
        'income': '',
        'spending': '',
        'age': '30',
        'gender': 'Female'
    }

    if request.method == 'POST':
        income_input = request.form.get('income', '').strip()
        spending_input = request.form.get('spending', '').strip()
        age_input = request.form.get('age', '30').strip()
        gender_input = request.form.get('gender', 'Female').strip()

        form_data = {
            'income': income_input,
            'spending': spending_input,
            'age': age_input,
            'gender': gender_input
        }

        # Validation
        errors = []
        try:
            income_val = float(income_input)
            if income_val < 1 or income_val > 500:
                errors.append("Annual Income must be a realistic value between $1k and $500k.")
        except ValueError:
            errors.append("Annual Income must be a valid number.")

        try:
            spending_val = float(spending_input)
            if spending_val < 1 or spending_val > 100:
                errors.append("Spending Score must be an integer between 1 and 100.")
        except ValueError:
            errors.append("Spending Score must be a valid number.")

        try:
            age_val = float(age_input)
            if age_val < 10 or age_val > 110:
                errors.append("Age must be between 10 and 110 years.")
        except ValueError:
            age_val = 35.0

        if errors:
            for err in errors:
                flash(err, "danger")
        elif kmeans_model is None or scaler is None:
            flash("Model artifacts are missing. Run train_model.py first.", "danger")
        else:
            # 1. K-Means Segment Prediction
            feature_cols = segment_metadata.get('kmeans_features', ['Annual Income (k$)', 'Spending Score (1-100)'])
            input_df = pd.DataFrame([[income_val, spending_val]], columns=feature_cols)
            scaled_features = scaler.transform(input_df)
            predicted_cluster = int(kmeans_model.predict(scaled_features)[0])

            # Retrieve segment details
            segment_details = segment_metadata['segments'].get(predicted_cluster, {})
            
            # Compute distance to centroid
            centroid_income = segment_details.get('centroid_income', income_val)
            centroid_spending = segment_details.get('centroid_spending', spending_val)
            dist_income = round(income_val - centroid_income, 1)
            dist_spending = round(spending_val - centroid_spending, 1)

            # 2. Logistic Regression Classification (High-Spending Propensity)
            log_result = None
            if logistic_model is not None and logistic_scaler is not None:
                gender_encoded = 1 if gender_input.lower() == 'male' else 0
                log_df = pd.DataFrame(
                    [[age_val, income_val, gender_encoded]],
                    columns=['Age', 'Annual Income (k$)', 'Gender_Encoded']
                )
                log_scaled = logistic_scaler.transform(log_df)
                high_spend_prob = float(logistic_model.predict_proba(log_scaled)[0, 1])
                is_high_spender = bool(high_spend_prob >= 0.5)

                log_result = {
                    "is_high_spender": is_high_spender,
                    "probability_pct": round(high_spend_prob * 100, 1),
                    "verdict": "High Spending Propensity" if is_high_spender else "Moderate / Conservative Spender"
                }

            prediction_result = {
                "cluster_id": predicted_cluster,
                "segment_name": segment_details.get('name', f'Cluster {predicted_cluster}'),
                "short_tag": segment_details.get('short_tag', 'Customer Segment'),
                "badge_color": segment_details.get('badge_color', '#2563eb'),
                "description": segment_details.get('description', ''),
                "strategy": segment_details.get('strategy', ''),
                "centroid_income": centroid_income,
                "centroid_spending": centroid_spending,
                "dist_income": dist_income,
                "dist_spending": dist_spending,
                "input_income": income_val,
                "input_spending": spending_val,
                "logistic_result": log_result
            }

    return render_template('prediction.html', prediction=prediction_result, form_data=form_data)


# =====================================================================
# API ENDPOINTS (For Interactive Visualizations)
# =====================================================================

@app.route('/api/customer-data')
def api_customer_data():
    """
    Returns clustered customer coordinates and centroid locations
    for Chart.js scatter rendering on the dashboard.
    """
    if df_customers is None or segment_metadata is None:
        return jsonify({"error": "Data not loaded"}), 404

    segments = segment_metadata.get('segments', {})
    datasets = []

    # One dataset per cluster
    for cluster_id in range(segment_metadata.get('n_clusters', 5)):
        cluster_df = df_customers[df_customers['Cluster'] == cluster_id]
        seg = segments.get(cluster_id, {})
        points = [
            {"x": row['Annual Income (k$)'], "y": row['Spending Score (1-100)'], "id": row['CustomerID']}
            for _, row in cluster_df.iterrows()
        ]
        datasets.append({
            "label": f"{seg.get('short_tag', f'Cluster {cluster_id}')}",
            "data": points,
            "backgroundColor": seg.get('badge_color', '#3b82f6') + 'cc',
            "borderColor": seg.get('badge_color', '#3b82f6'),
            "pointRadius": 5,
            "pointHoverRadius": 8
        })

    # Centroids dataset
    centroid_points = [
        {"x": seg['centroid_income'], "y": seg['centroid_spending'], "label": f"C{cid}"}
        for cid, seg in segments.items()
    ]
    datasets.append({
        "label": "Cluster Centroids",
        "data": centroid_points,
        "backgroundColor": "#dc2626",
        "borderColor": "#1e293b",
        "borderWidth": 2,
        "pointStyle": 'star',
        "pointRadius": 12,
        "pointHoverRadius": 14
    })

    return jsonify({
        "datasets": datasets,
        "total": len(df_customers)
    })


if __name__ == '__main__':
    # Run development server locally
    app.run(host='127.0.0.1', port=5000, debug=True)
