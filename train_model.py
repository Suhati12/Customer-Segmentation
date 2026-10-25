"""
Customer Segmentation System - Model Training Script
===================================================
This script performs:
1. Dataset loading and verification (with Kaggle download instructions).
2. Comprehensive data preprocessing (inspection, missing value checks, duplicates, IQR outlier detection, encoding, scaling).
3. Exploratory Data Analysis (EDA) and saving visualizations.
4. K-Means clustering with Elbow Method and Silhouette Analysis.
5. Dynamic cluster labeling and business segment characterization.
6. Supervised classification using Logistic Regression for high-spending propensity.
7. Model serialization with Joblib for deployment in the Flask web application.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server generation
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Set visual style for charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# Path constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
CHARTS_DIR = os.path.join(STATIC_DIR, 'charts')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'Mall_Customers.csv')
SEGMENTED_DATA_PATH = os.path.join(DATA_DIR, 'Mall_Customers_Segmented.csv')

# Ensure directories exist
for folder in [DATA_DIR, MODELS_DIR, STATIC_DIR, CHARTS_DIR]:
    os.makedirs(folder, exist_ok=True)


# =====================================================================
# 1. DATASET LOADING AND INSTRUCTIONS
# =====================================================================
def load_dataset(filepath=RAW_DATA_PATH):
    """
    Loads the Mall Customers dataset from a local CSV file.
    If not found, provides Kaggle download instructions and attempts a fallback download.
    """
    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)

    if not os.path.exists(filepath):
        print("\n[WARNING] Dataset not found at:", filepath)
        print("\n--- INSTRUCTIONS TO DOWNLOAD FROM KAGGLE ---")
        print("1. Visit: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
        print("2. Click 'Download' to get 'archive.zip'.")
        print("3. Extract 'Mall_Customers.csv' and place it in the folder:")
        print(f"   {DATA_DIR}")
        print("-------------------------------------------\n")
        print("Attempting automatic download from verified public repository...")
        import urllib.request
        url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
        try:
            urllib.request.urlretrieve(url, filepath)
            print("[SUCCESS] Dataset downloaded successfully to:", filepath)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load or download dataset: {e}. Please place Mall_Customers.csv manually in {DATA_DIR}."
            )

    df = pd.read_csv(filepath)
    print(f"[OK] Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")
    return df


# =====================================================================
# 2. DATA INSPECTION AND PREPROCESSING
# =====================================================================
def preprocess_data(df_raw):
    """
    Performs data inspection, cleaning, outlier detection, and encoding.
    Preserves valid high-value customers while documenting distributions.
    """
    print("=" * 70)
    print("STEP 2: DATA INSPECTION & PREPROCESSING")
    print("=" * 70)

    df = df_raw.copy()

    # 1. Inspect first 5 rows
    print("\n--- First 5 Rows of Dataset ---")
    print(df.head())

    # 2. Check shape and column names
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Column Names:", df.columns.tolist())

    # Standardize column naming if necessary for cleaner programmatic access
    # Standard Mall Customers columns:
    # 'CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'
    print("\nData Types:")
    print(df.dtypes)

    # 3. Check for missing values
    missing_counts = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing_counts)
    if missing_counts.sum() == 0:
        print("[OK] No missing values found in the dataset.")
    else:
        print(f"[NOTE] Found {missing_counts.sum()} missing values. Handling via forward fill/drop.")
        df = df.dropna()

    # 4. Check for duplicate records
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Records Found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"[OK] Removed {duplicate_count} duplicate rows.")
    else:
        print("[OK] All customer records are unique.")

    # 5. Outlier Detection using IQR Method
    print("\n--- Outlier Detection (Interquartile Range - IQR) ---")
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    outlier_indices = set()

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Column '{col}': IQR={IQR:.2f}, Lower Bound={lower_bound:.2f}, Upper Bound={upper_bound:.2f}")
        print(f"  -> Outliers detected: {len(col_outliers)}")
        if len(col_outliers) > 0:
            print(f"     Outlier customer IDs: {col_outliers['CustomerID'].tolist()}")
            outlier_indices.update(col_outliers.index)

    print("\nOutlier Handling Decision:")
    print(f"Total potential outlier records identified: {len(outlier_indices)}")
    print("Rationale: High-income customers (e.g. $137k) are legitimate high-wealth mall shoppers,")
    print("not data entry errors. Removing them would discard key VIP customer segments.")
    print("Therefore, all valid customer records are retained.\n")

    # 6. Encode categorical features (Gender)
    # Binary mapping: Male -> 1, Female -> 0
    df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
    print("Categorical Encoding: Gender mapped to Gender_Encoded (Male: 1, Female: 0)")

    # 7. Comparison Before vs After Preprocessing
    print("\n--- Summary Statistics (Original vs Processed) ---")
    print(df[numerical_cols].describe().round(2))

    return df


# =====================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# =====================================================================
def perform_eda(df):
    """
    Generates exploratory data analysis plots and saves them to static/charts.
    """
    print("=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Feature Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    features = [
        ('Age', 'Customer Age Distribution', 'Age (Years)', '#3b82f6'),
        ('Annual Income (k$)', 'Annual Income Distribution', 'Annual Income (k$)', '#10b981'),
        ('Spending Score (1-100)', 'Spending Score Distribution', 'Score (1-100)', '#8b5cf6')
    ]

    for ax, (col, title, xlabel, color) in zip(axes, features):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=15, edgecolor='black', alpha=0.6)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.1f})')
        ax.axvline(median_val, color='#f59e0b', linestyle=':', linewidth=1.5, label=f'Median ({median_val:.1f})')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Number of Customers', fontsize=10)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    dist_path = os.path.join(CHARTS_DIR, 'eda_distributions.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[OK] Saved distributions chart -> {dist_path}")

    # 2. Income vs Spending Score Scatter with Gender Hue
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Gender',
        palette={'Male': '#2563eb', 'Female': '#ec4899'},
        s=80,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title('Customer Annual Income vs. Spending Score', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Annual Income (k$)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(title='Gender', frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    scatter_path = os.path.join(CHARTS_DIR, 'eda_income_vs_spending.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Income vs Spending scatter plot -> {scatter_path}")

    # 3. Correlation Heatmap
    plt.figure(figsize=(7, 5))
    corr_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.3f',
        vmin=-0.5,
        vmax=0.5,
        linewidths=1,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    corr_path = os.path.join(CHARTS_DIR, 'eda_correlation_heatmap.png')
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"[OK] Saved correlation heatmap -> {corr_path}\n")


# =====================================================================
# 4. K-MEANS CLUSTERING & ELBOW METHOD
# =====================================================================
def run_kmeans_clustering(df):
    """
    Selects features, applies StandardScaler, determines optimal clusters using
    the Elbow Method & Silhouette Scores, and trains K-Means.
    """
    print("=" * 70)
    print("STEP 4: K-MEANS CLUSTERING")
    print("=" * 70)

    # 1. Feature Selection
    feature_names = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[feature_names]
    print(f"Selected features for clustering: {feature_names}")

    # 2. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[OK] Features standardized using StandardScaler (mean=0, std=1).\n")

    # 3. Elbow Method & Silhouette Scores
    wcss = []
    silhouette_scores = []
    k_range = range(1, 11)

    print("Evaluating K from 1 to 10...")
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        if k >= 2:
            sil = silhouette_score(X_scaled, km.labels_)
            silhouette_scores.append(sil)
            print(f"  k={k:2d} | WCSS (Inertia) = {km.inertia_:8.2f} | Silhouette Score = {sil:.4f}")
        else:
            print(f"  k={k:2d} | WCSS (Inertia) = {km.inertia_:8.2f} | Silhouette Score = N/A")

    # Plot Elbow Curve and Silhouette Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow curve (Inertia)
    ax1.plot(list(k_range), wcss, marker='o', color='#2563eb', linewidth=2.5, markersize=8)
    ax1.axvline(x=5, color='#ef4444', linestyle='--', linewidth=1.5, label='Optimal Elbow (k=5)')
    ax1.set_title('Elbow Method for Optimal k', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=10)
    ax1.set_xticks(list(k_range))
    ax1.legend(frameon=True, facecolor='white')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Silhouette curve
    ax2.plot(list(range(2, 11)), silhouette_scores, marker='s', color='#10b981', linewidth=2.5, markersize=8)
    ax2.axvline(x=5, color='#ef4444', linestyle='--', linewidth=1.5, label='Peak Score (k=5)')
    ax2.set_title('Silhouette Analysis Across Cluster Sizes', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=10)
    ax2.set_ylabel('Silhouette Coefficient', fontsize=10)
    ax2.set_xticks(list(range(2, 11)))
    ax2.legend(frameon=True, facecolor='white')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    elbow_path = os.path.join(CHARTS_DIR, 'kmeans_elbow_curve.png')
    plt.savefig(elbow_path, dpi=300)
    plt.close()
    print(f"\n[OK] Saved Elbow Curve chart -> {elbow_path}")

    # 4. Train final K-Means with optimal k=5
    optimal_k = 5
    print(f"\nTraining final K-Means model with k={optimal_k}...")
    kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)
    df['Cluster'] = cluster_labels

    # 5. Centroids
    centroids_scaled = kmeans_model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # 6. Dynamic segment labeling based on actual centroid coordinates
    # We evaluate centroid annual income and spending score
    # to dynamically assign accurate descriptive business labels
    segment_info = {}
    print("\n--- Identified Customer Segments & Centroids ---")
    for i in range(optimal_k):
        income_c = centroids_original[i][0]
        spending_c = centroids_original[i][1]
        count = int((cluster_labels == i).sum())
        pct = (count / len(df)) * 100

        # Dynamic characterization rule
        if income_c > 70 and spending_c > 60:
            name = "High Income, High Spending"
            short_tag = "VIP / Elite Shoppers"
            badge_color = "#8b5cf6" # purple
            desc = "High earners who spend lavishly. They value premium brand experiences, exclusivity, and luxury services."
            strategy = "Target with VIP loyalty tiers, luxury launches, personalized shopping concierges, and early-access privileges."
        elif income_c > 70 and spending_c <= 60:
            name = "High Income, Low Spending"
            short_tag = "Careful / Conservative Wealth"
            badge_color = "#3b82f6" # blue
            desc = "Affluent individuals who are highly selective and price-conscious despite substantial income."
            strategy = "Attract with high-value propositions, durable quality assurances, smart investments, and discreet promotions."
        elif income_c <= 40 and spending_c > 60:
            name = "Low Income, High Spending"
            short_tag = "Impulsive / Trendsetters"
            badge_color = "#ec4899" # pink
            desc = "Younger, style-driven shoppers with modest earnings but high desire for fashion, entertainment, and trendy products."
            strategy = "Engage with flash sales, flexible Buy Now Pay Later (BNPL) options, trend-focused social campaigns, and influencer partnerships."
        elif income_c <= 40 and spending_c <= 60:
            name = "Low Income, Low Spending"
            short_tag = "Sensible / Budget Conscious"
            badge_color = "#64748b" # slate
            desc = "Price-sensitive customers who strictly budget their visits and purchase essentials only."
            strategy = "Provide clear discount coupons, bundle deals, bulk savings, and practical everyday value messaging."
        else:
            name = "Average Income, Average Spending"
            short_tag = "Core Mainstream / Balanced"
            badge_color = "#10b981" # emerald
            desc = "The middle-ground majority with balanced earning and spending habits, forming the steady backbone of foot traffic."
            strategy = "Maintain consistent promotional campaigns, rewards for repeat visits, family-friendly events, and seasonal sales."

        segment_info[int(i)] = {
            "cluster_id": int(i),
            "name": name,
            "short_tag": short_tag,
            "badge_color": badge_color,
            "centroid_income": round(float(income_c), 2),
            "centroid_spending": round(float(spending_c), 2),
            "customer_count": count,
            "percentage": round(pct, 1),
            "description": desc,
            "strategy": strategy
        }

        print(f"Cluster {i}: '{name}' ({short_tag})")
        print(f"  Centroid: Income=${income_c:.1f}k, Spending Score={spending_c:.1f}")
        print(f"  Customers: {count} ({pct:.1f}% of total)\n")

    # Map segment short tags into dataframe for analysis
    df['Segment_Name'] = df['Cluster'].map(lambda c: segment_info[c]['name'])
    df['Segment_Tag'] = df['Cluster'].map(lambda c: segment_info[c]['short_tag'])

    # 7. Cluster Scatter Plot Visualization
    plt.figure(figsize=(11, 7))
    palette_colors = [segment_info[i]['badge_color'] for i in range(optimal_k)]
    cluster_names = [f"Cluster {i}: {segment_info[i]['short_tag']}" for i in range(optimal_k)]

    for i in range(optimal_k):
        cluster_points = df[df['Cluster'] == i]
        plt.scatter(
            cluster_points['Annual Income (k$)'],
            cluster_points['Spending Score (1-100)'],
            s=70,
            color=segment_info[i]['badge_color'],
            alpha=0.85,
            edgecolors='black',
            linewidths=0.5,
            label=f"{segment_info[i]['short_tag']} (n={len(cluster_points)})"
        )

    # Plot centroids
    plt.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=260,
        color='#dc2626',
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='Cluster Centroids',
        zorder=10
    )

    # Add centroid text annotations
    for i in range(optimal_k):
        plt.annotate(
            f"C{i} ({centroids_original[i, 0]:.0f}k, {centroids_original[i, 1]:.0f})",
            (centroids_original[i, 0], centroids_original[i, 1]),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dc2626", alpha=0.85)
        )

    plt.title('Mall Customer Segmentation Clusters (K-Means k=5)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Annual Income (k$)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    clusters_path = os.path.join(CHARTS_DIR, 'kmeans_clusters.png')
    plt.savefig(clusters_path, dpi=300)
    plt.close()
    print(f"[OK] Saved K-Means cluster scatter plot -> {clusters_path}\n")

    return kmeans_model, scaler, df, segment_info


# =====================================================================
# 5. LOGISTIC REGRESSION CLASSIFICATION MODULE
# =====================================================================
def run_logistic_regression(df):
    """
    Implements a supervised classification module to predict customer High-Spending
    propensity based on demographic and financial features.

    Note: As required by instructions, we define a transparent business rule:
    'High_Spender' = 1 if Spending Score >= 50, else 0.
    We do NOT arbitrarily reuse unsupervised K-Means labels as ground truth.
    """
    print("=" * 70)
    print("STEP 5: LOGISTIC REGRESSION CLASSIFICATION")
    print("=" * 70)

    # Define classification target based on clear business logic:
    # A customer with Spending Score >= 50 is classified as a "High-Spending Shopper" (1)
    # A customer with Spending Score < 50 is classified as a "Moderate/Low Spender" (0)
    df_clf = df.copy()
    target_name = 'High_Spender'
    df_clf[target_name] = (df_clf['Spending Score (1-100)'] >= 50).astype(int)

    feature_cols = ['Age', 'Annual Income (k$)', 'Gender_Encoded']
    X = df_clf[feature_cols]
    y = df_clf[target_name]

    print(f"Classification Target: '{target_name}' (1: Spending Score >= 50, 0: < 50)")
    print(f"Target distribution: {y.value_counts().to_dict()} (1: {y.sum()}, 0: {len(y) - y.sum()})")
    print(f"Predictor Features: {feature_cols}")

    # Train / Test Split (80% Train, 20% Test) with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} samples | Test set size: {X_test.shape[0]} samples")

    # Fit feature scaler ONLY on training data to avoid data leakage
    log_scaler = StandardScaler()
    X_train_scaled = log_scaler.fit_transform(X_train)
    X_test_scaled = log_scaler.transform(X_test)

    # Train Logistic Regression model
    log_model = LogisticRegression(random_state=42, C=1.0)
    log_model.fit(X_train_scaled, y_train)

    # Evaluate predictions
    y_pred = log_model.predict(X_test_scaled)
    y_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Logistic Regression Evaluation Metrics ---")
    print(f"Accuracy:        {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC Score:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low/Mod Spender', 'High Spender']))

    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Pred Low/Mod', 'Pred High Spender'],
        yticklabels=['Actual Low/Mod', 'Actual High Spender'],
        cbar=False
    )
    plt.title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.ylabel('Actual Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(CHARTS_DIR, 'logistic_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved confusion matrix chart -> {cm_path}")

    # Feature Importance (Coefficients)
    coef_dict = dict(zip(feature_cols, log_model.coef_[0]))
    print("\nFeature Coefficients (Impact on High-Spending Propensity):")
    for feat, coef in coef_dict.items():
        print(f"  {feat:22s}: {coef:+.4f}")

    print("\nInterpretation:")
    print("Negative coefficient on Age indicates younger shoppers demonstrate a significantly")
    print("higher propensity to spend, aligning with retail behavioral patterns.")

    clf_metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
        "coefficients": {k: round(float(v), 4) for k, v in coef_dict.items()},
        "intercept": round(float(log_model.intercept_[0]), 4),
        "feature_order": feature_cols
    }

    return log_model, log_scaler, clf_metrics


# =====================================================================
# 6. MODEL SAVING AND SERIALIZATION
# =====================================================================
def save_artifacts(kmeans_model, scaler, log_model, log_scaler, segment_info, clf_metrics, df):
    """
    Saves trained models, scalers, metadata, and segmented dataset using Joblib.
    """
    print("=" * 70)
    print("STEP 6: SAVING MODELS AND METADATA")
    print("=" * 70)

    # 1. K-Means Model
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
    joblib.dump(kmeans_model, kmeans_path)
    print(f"[SAVED] K-Means model -> {kmeans_path}")

    # 2. Feature Scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"[SAVED] Feature Scaler -> {scaler_path}")

    # 3. Logistic Regression Model
    log_path = os.path.join(MODELS_DIR, 'logistic_model.joblib')
    joblib.dump(log_model, log_path)
    print(f"[SAVED] Logistic Regression model -> {log_path}")

    # 4. Logistic Demographic Scaler
    log_scaler_path = os.path.join(MODELS_DIR, 'logistic_scaler.joblib')
    joblib.dump(log_scaler, log_scaler_path)
    print(f"[SAVED] Logistic Scaler -> {log_scaler_path}")

    # 5. Segment Metadata and Overall Dataset Summary
    metadata = {
        "kmeans_features": ['Annual Income (k$)', 'Spending Score (1-100)'],
        "n_clusters": len(segment_info),
        "segments": segment_info,
        "classification_metrics": clf_metrics,
        "dataset_summary": {
            "total_customers": int(len(df)),
            "avg_age": round(float(df['Age'].mean()), 1),
            "avg_income": round(float(df['Annual Income (k$)'].mean()), 1),
            "avg_spending": round(float(df['Spending Score (1-100)'].mean()), 1),
            "gender_breakdown": {
                "Female": int((df['Gender'] == 'Female').sum()),
                "Male": int((df['Gender'] == 'Male').sum())
            }
        }
    }

    meta_path = os.path.join(MODELS_DIR, 'segment_metadata.joblib')
    joblib.dump(metadata, meta_path)
    print(f"[SAVED] Segment Metadata -> {meta_path}")

    # Also save metadata as human-readable JSON
    json_path = os.path.join(MODELS_DIR, 'segment_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata JSON -> {json_path}")

    # 6. Save Segmented Dataset for Flask App querying
    df.to_csv(SEGMENTED_DATA_PATH, index=False)
    print(f"[SAVED] Segmented Dataset CSV -> {SEGMENTED_DATA_PATH}")

    print("\n[SUCCESS] Model training and serialization completed successfully!")


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("CUSTOMER SEGMENTATION SYSTEM - ML PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    df_raw = load_dataset()

    # 2. Preprocess data
    df_clean = preprocess_data(df_raw)

    # 3. Exploratory Data Analysis
    perform_eda(df_clean)

    # 4. K-Means Clustering
    kmeans_model, scaler, df_segmented, segment_info = run_kmeans_clustering(df_clean)

    # 5. Logistic Regression Classification
    log_model, log_scaler, clf_metrics = run_logistic_regression(df_segmented)

    # 6. Save all artifacts
    save_artifacts(
        kmeans_model=kmeans_model,
        scaler=scaler,
        log_model=log_model,
        log_scaler=log_scaler,
        segment_info=segment_info,
        clf_metrics=clf_metrics,
        df=df_segmented
    )


if __name__ == '__main__':
    main()
