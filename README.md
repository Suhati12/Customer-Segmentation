# Customer Segmentation System

An end-to-end Machine Learning and Web Application for retail customer segmentation using **K-Means Clustering**, **Logistic Regression Classification**, and a modern **Flask Web Application**.

![Customer Segmentation Preview](static/charts/kmeans_clusters.png)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Technologies Required](#technologies-required)
- [Dataset & Kaggle Download Instructions](#dataset--kaggle-download-instructions)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Machine Learning Workflow](#machine-learning-workflow)
  - [1. Data Preprocessing & Outlier Handling](#1-data-preprocessing--outlier-handling)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. K-Means Clustering & Optimal k](#3-k-means-clustering--optimal-k)
  - [4. Customer Segments & Marketing Strategies](#4-customer-segments--marketing-strategies)
  - [5. Supervised Customer Classification (Logistic Regression)](#5-supervised-customer-classification-logistic-regression)
- [Web Application Walkthrough](#web-application-walkthrough)
- [Running the Experiments (Jupyter Notebook)](#running-the-experiments-jupyter-notebook)
- [Results & Key Findings](#results--key-findings)

---

## 🎯 Project Overview
In modern retail, treating all customers uniformly results in inefficient marketing spend, lower conversion rates, and churn. **Customer Segmentation** partitions consumers into distinct, actionable cohorts based on purchasing behaviors and financial capacity.

This project delivers:
1. **Unsupervised Clustering**: K-Means clustering ($k=5$) identifying natural customer groupings.
2. **Supervised Spend Propensity**: Logistic Regression model scoring prospective shoppers on their likelihood of being a high spender ($\ge 50$).
3. **Interactive Web Application**: A responsive Flask web application featuring an executive KPI dashboard, dynamic Chart.js visualizations, customer search & filtering, and a real-time segment prediction tool.
4. **Research Notebook**: A Jupyter notebook documenting every phase of data preprocessing, mathematical clustering, and classification.

---

## 🛠️ Technologies Required
- **Core Language**: Python (3.10+)
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn, Chart.js
- **Machine Learning**: Scikit-learn (KMeans, StandardScaler, LogisticRegression, Silhouette Score, Train-Test Split)
- **Model Serialization**: Joblib
- **Web Backend**: Flask
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphic Dark Theme, Responsive Flex/Grid)

---

## 📊 Dataset & Kaggle Download Instructions
The system uses the benchmark **Mall Customers Dataset**, containing 200 visitor records:
- **CustomerID**: Unique identifier (1 to 200).
- **Gender**: Customer biological sex (`Male` / `Female`).
- **Age**: Age in years (18 to 70).
- **Annual Income (k$)**: Annual earnings in thousands of dollars ($15k to $137k).
- **Spending Score (1-100)**: Proprietary mall behavioral index reflecting customer purchase frequency and expenditure.

### Instructions to Download from Kaggle
If the dataset is not yet present on your machine:
1. Open your browser and navigate to the Kaggle dataset page:  
   👉 **[Customer Segmentation Tutorial in Python on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)**
2. Click the **Download** button (requires a free Kaggle account).
3. Extract `Mall_Customers.csv` from the downloaded archive.
4. Place `Mall_Customers.csv` into the `data/` directory:
   ```bash
   customer_segmentation/data/Mall_Customers.csv
   ```
*(Note: If the file is missing when running `train_model.py`, the script will automatically attempt a verified fallback download).*

---

## 📂 Project Structure
```
customer_segmentation/
│
├── app.py                     # Flask web server and routing logic
├── train_model.py             # Preprocessing, ML model training, and serialization
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation and execution guide
├── REPORT.md                  # Short academic and engineering project report
│
├── data/
│   ├── Mall_Customers.csv            # Original raw dataset
│   └── Mall_Customers_Segmented.csv  # Cleaned dataset with cluster assignments
│
├── models/
│   ├── kmeans_model.joblib      # Trained K-Means clustering model (k=5)
│   ├── scaler.joblib            # Fitted StandardScaler for income & spending
│   ├── logistic_model.joblib    # Trained Logistic Regression classifier
│   ├── logistic_scaler.joblib   # Fitted StandardScaler for demographic features
│   ├── segment_metadata.joblib  # Serialized segment labels, centroids & statistics
│   └── segment_metadata.json    # Human-readable JSON metadata
│
├── notebooks/
│   └── customer_segmentation.ipynb   # Complete Jupyter Notebook workflow
│
├── templates/
│   ├── base.html              # Shared layout template with navigation & alerts
│   ├── index.html             # Landing page with introduction & segment preview
│   ├── dashboard.html         # Executive KPI dashboard & interactive charts
│   ├── customers.html         # Customer records search, filter & characteristics
│   └── prediction.html        # Real-time customer segment prediction form
│
└── static/
    ├── style.css              # Custom responsive stylesheet (Dark Glassmorphism)
    └── charts/                # Pre-rendered high-res Matplotlib visualization exports
        ├── eda_distributions.png
        ├── eda_income_vs_spending.png
        ├── eda_correlation_heatmap.png
        ├── kmeans_elbow_curve.png
        ├── kmeans_clusters.png
        └── logistic_confusion_matrix.png
```

---

## 🚀 Installation & Setup

### 1. Clone or Navigate to the Project Folder
```bash
cd customer_segmentation
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Models
Execute the training script to perform preprocessing, compute the Elbow curve, train K-Means and Logistic Regression, generate high-resolution visualization charts, and save the model artifacts:
```bash
python train_model.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing & Outlier Handling
- **Data Inspection**: Verified 200 records with 5 columns.
- **Null & Duplicate Checks**: 0 missing values, 0 duplicate records.
- **Outlier Detection via IQR Method**:
  - $Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$ bounds calculated for `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.
  - Only two records (Customer IDs 199 and 200) exceeded the statistical threshold on Annual Income ($137k).
  - **Decision**: These individuals represent authentic affluent shoppers. Discarding them would eliminate our luxury/VIP segment. All 200 records were preserved.
- **Categorical Encoding**: `Gender` encoded to binary `Gender_Encoded` (`Male`: 1, `Female`: 0).
- **Feature Standardization**: Transformed `Annual Income (k$)` and `Spending Score (1-100)` to zero mean and unit variance ($\mu=0, \sigma=1$) using `StandardScaler` to prevent distance bias in Euclidean space.

### 2. Exploratory Data Analysis (EDA)
- **Age vs. Spending**: Pearson correlation of $-0.327$ demonstrates that younger consumers have a notably higher mall spending propensity.
- **Income vs. Spending**: Linear correlation is near-zero ($r = 0.0099$), confirming that customer spending is non-linear and demands geometric spatial clustering.
- **Distributions**: Median Age is 36.0 years, Median Income is $61.5k, and Median Spending Score is 50.0.

### 3. K-Means Clustering & Optimal k
- Evaluated $k$ from 1 to 10 using Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette Analysis.
- **Elbow Point**: Distinct leveling-off observed at $k = 5$.
- **Silhouette Coefficient**: Peaks at **0.5547** at $k = 5$ (compared to 0.4939 at $k=4$ and 0.5399 at $k=6$).
- Model trained with `KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)`.

### 4. Customer Segments & Marketing Strategies

| Cluster | Segment Name | Short Tag | Centroid Income | Centroid Spend | Share | Actionable Marketing Strategy |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | Average Income, Average Spending | **Core Mainstream / Balanced** | $55.3k | 49.5 / 100 | **40.5%** | Consistent promotional campaigns, seasonal sales, and repeat-visit loyalty points. |
| **1** | High Income, High Spending | **VIP / Elite Shoppers** | $86.5k | 82.1 / 100 | **19.5%** | VIP loyalty tiers, luxury brand launches, private concierges, and early-access privileges. |
| **2** | Low Income, High Spending | **Impulsive / Trendsetters** | $25.7k | 79.4 / 100 | **11.0%** | Flash sales, Buy Now Pay Later (BNPL) options, trend-focused social media campaigns. |
| **3** | High Income, Low Spending | **Careful / Conservative Wealth** | $88.2k | 17.1 / 100 | **17.5%** | High-value propositions, durable product quality assurances, and smart financial messaging. |
| **4** | Low Income, Low Spending | **Sensible / Budget Conscious** | $26.3k | 20.9 / 100 | **11.5%** | Clear discount vouchers, everyday value bundles, and essentials reward programs. |

### 5. Supervised Customer Classification (Logistic Regression)
To score prospective customers before comprehensive spending tracking is available:
- **Business Target**: Defined as `High_Spender` = 1 if $\text{Spending Score} \ge 50$ (balanced 51% / 49% distribution).
- **Features**: `Age`, `Annual Income (k$)`, `Gender_Encoded`.
- **Validation**: 80% train, 20% test stratified split; scaler fitted strictly on training data to prevent data leakage.
- **Test Metrics**:
  - **Accuracy**: **72.5%**
  - **Precision**: **0.737**
  - **Recall**: **0.700**
  - **F1-Score**: **0.718**
  - **ROC-AUC**: **0.680**
- **Insight**: `Age` had a strong negative coefficient ($-0.591$), corroborating that younger visitors have higher odds of active expenditure.

---

## 🌐 Web Application Walkthrough

### 1. Home Page (`/`)
- Professional introductory overview of machine learning customer segmentation.
- Quick statistical highlights (200 shoppers, 5 segments, $60.6k avg income).
- Interactive cards describing each of the 5 discovered segments.

### 2. Executive Dashboard (`/dashboard`)
- 4 real-time KPI metric summary cards.
- **Interactive Chart.js Scatter Plot**: Hover over any point to inspect customer ID, income, and spending score. Centroids are marked with prominent red stars.
- Toggle between dynamic Chart.js canvas and high-resolution Matplotlib plot.
- Demographic distribution charts (Gender breakdown doughnut & Age bracket bar chart).
- Preprocessing and validation gallery (Elbow curve, KDE distributions, Correlation heatmap).
- Supervised Logistic Regression performance card with confusion matrix image.

### 3. Customer Analysis View (`/customers`)
- Search any individual customer by numeric ID.
- Filter the entire customer database by specific cluster segment.
- Interactive table showing Customer ID, Gender, Age, Income, Spending Score, and colored Segment Badges.
- Comprehensive segment characteristics reference drawer.

### 4. Real-Time Customer Segment Predictor (`/predict`)
- Form accepting Annual Income ($k), Spending Score (1-100), Age, and Gender.
- Comprehensive server-side validation with friendly error notifications.
- Normalizes inputs using the pre-fitted `StandardScaler` and queries `KMeans`.
- Outputs:
  - Exact Segment Name and Custom Color Badge.
  - Centroid distance analysis (income & spending offsets).
  - Detailed customer persona summary.
  - Actionable business/marketing strategy.
  - High-Spending Propensity probability gauge from the Logistic Regression model.

---

## 📓 Running the Experiments (Jupyter Notebook)
To inspect the step-by-step experimental workflow in Jupyter:
```bash
jupyter notebook notebooks/customer_segmentation.ipynb
```
The notebook contains full markdown explanations, visualizations, mathematical formulas, and step-by-step code outputs.

---

## ⚖️ Limitations & Next Steps
1. **Dataset Scope**: The Mall Customers dataset comprises 200 records without temporal transaction history.
2. **Next Steps**:
   - Integrating RFM (Recency, Frequency, Monetary) metrics.
   - Experimenting with Density-Based Clustering (DBSCAN) or Hierarchical Agglomerative Clustering for non-convex cluster shapes.
