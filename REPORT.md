# Customer Segmentation System: Project Report

**Author:** Machine Learning Engineering Team  
**Dataset:** Mall Customers Dataset (Kaggle)  
**System Architecture:** Python, Scikit-learn, K-Means Clustering, Logistic Regression, Flask Web App

---

## 1. Problem Statement
Retail stores and shopping malls serve heterogeneous populations with widely divergent financial capacities, lifestyle preferences, and shopping motivations. When retailers apply generalized, one-size-fits-all promotional campaigns:
- Marketing budgets are wasted on unresponsive demographics.
- High-income, high-spending patrons are not offered personalized luxury incentives, leading to missed revenue opportunities.
- Budget-conscious customers are alienated by high-ticket offers, driving churn.

To maximize marketing return-on-investment (ROI) and optimize customer retention, mall management requires an automated, data-driven system to partition customer records into actionable segments and score new visitors in real time.

---

## 2. Project Objectives
1. **Data Preprocessing & Cleaning**: Inspect and clean the Mall Customers dataset, detect outliers via the Interquartile Range (IQR) method while preserving authentic high-net-worth customers, encode categorical variables, and apply feature standardization.
2. **Exploratory Data Analysis (EDA)**: Identify demographic and financial relationships (Age, Gender, Income, Spending Score) through descriptive visualizations.
3. **Unsupervised Customer Segmentation**: Implement K-Means clustering, determine the mathematically optimal number of clusters using the Elbow Method and Silhouette Analysis, calculate cluster centroids, and establish actionable business profiles.
4. **Supervised Customer Classification**: Develop an independent Logistic Regression model to predict high-spending propensity ($\ge 50$) from demographic inputs, evaluating accuracy, precision, recall, F1-score, and confusion matrix.
5. **Full-Stack Web Application**: Deliver a responsive, glassmorphic Flask web platform featuring an executive dashboard, interactive Chart.js visualizations, customer search/filter analysis, and a real-time segment prediction tool.
6. **Reproducibility**: Package trained models with Joblib, create an interactive Jupyter Notebook, and provide complete documentation.

---

## 3. Methodology & Preprocessing

### 3.1 Data Acquisition & Inspection
The dataset contains 200 records across 5 attributes: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.
- Missing value audit confirmed zero null values across all columns.
- Duplicate detection confirmed 200 unique customer entities.

### 3.2 Outlier Analysis (IQR Method)
Using the Interquartile Range ($IQR = Q_3 - Q_1$), thresholds were computed:
- `Age`: Bounds $[-1.62, 79.38]$ &rarr; 0 outliers.
- `Spending Score`: Bounds $[-22.62, 130.38]$ &rarr; 0 outliers.
- `Annual Income`: Bounds $[-13.25, 132.75]$ &rarr; 2 records identified ($137k income, Customer IDs 199 and 200).

**Methodological Decision**: These two records represent legitimate affluent shoppers rather than faulty telemetry. Discarding them would truncate the upper bound of the customer income distribution and distort the luxury cluster. Therefore, all 200 records were preserved.

### 3.3 Feature Transformation & Scaling
- `Gender` was mapped into binary indicator `Gender_Encoded` (`Male`: 1, `Female`: 0).
- For K-Means clustering, `Annual Income (k$)` and `Spending Score (1-100)` were standardized to zero mean and unit variance using `StandardScaler` to prevent unequal distance weighting in Euclidean metric space:

$$z = \frac{x - \mu}{\sigma}$$

---

## 4. Algorithms & Mathematical Formulations

### 4.1 K-Means Clustering
K-Means partitions $N$ observations into $K$ mutually exclusive clusters by minimizing Within-Cluster Sum of Squares (Inertia):

$$\text{WCSS} = \sum_{j=1}^{K} \sum_{x \in C_j} ||x - \mu_j||^2$$

where $\mu_j$ is the centroid of cluster $C_j$.

#### Optimal Cluster Selection
1. **The Elbow Method**: Evaluated $k \in [1, 10]$. Inertia decreased sharply from $400.00$ ($k=1$) to $65.57$ ($k=5$), after which rate of decrease plateaued ($55.06$ at $k=6$).
2. **Silhouette Coefficient**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   - $k=2$: $0.3213$
   - $k=3$: $0.4666$
   - $k=4$: $0.4939$
   - **$k=5$**: **$0.5547$ (Global Maximum)**
   - $k=6$: $0.5399$
   - $k=7$: $0.5281$

Both criteria converge on **$k = 5$** as the optimal clustering configuration.

### 4.2 Supervised Logistic Regression Classifier
To evaluate prospective customer spend propensity without pre-existing spending logs:
- **Target Formulation**: `High_Spender` = 1 if $\text{Spending Score} \ge 50$, else 0.
- **Features**: `Age`, `Annual Income (k$)`, `Gender_Encoded`.
- **Validation**: 80/20 train/test stratified split ($N_{\text{train}} = 160, N_{\text{test}} = 40$). The scaler was fit exclusively on the training set to prevent data leakage.
- **Hypothesis Function**:
  $$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Income} + \beta_3 \cdot \text{Gender})}}$$

---

## 5. Results & Discussion

### 5.1 Customer Segments Identified ($k=5$)

| Cluster | Segment Name | Short Tag | Centroid Income | Centroid Spend | Count (Share) | Business Profile & Marketing Activation |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | Average Income, Average Spending | **Core Mainstream / Balanced** | $55.3k | 49.5 / 100 | 81 (40.5%) | Represents the stable traffic backbone. Strategy: Seasonal promotions, family events, and repeat purchase rewards. |
| **1** | High Income, High Spending | **VIP / Elite Shoppers** | $86.5k | 82.1 / 100 | 39 (19.5%) | Prime revenue drivers with high purchasing power. Strategy: Luxury tiers, early access, personal concierges. |
| **2** | Low Income, High Spending | **Impulsive / Trendsetters** | $25.7k | 79.4 / 100 | 22 (11.0%) | Younger demographic keen on fashion and lifestyle. Strategy: Buy Now Pay Later (BNPL), social campaigns, flash sales. |
| **3** | High Income, Low Spending | **Careful / Conservative Wealth** | $88.2k | 17.1 / 100 | 35 (17.5%) | Affluent but selective shoppers. Strategy: Quality warranties, premium durability messaging, financial value appeals. |
| **4** | Low Income, Low Spending | **Sensible / Budget Conscious** | $26.3k | 20.9 / 100 | 23 (11.5%) | Frugal, budget-limited visitors. Strategy: Clear discount vouchers, bulk savings, essential rewards. |

### 5.2 Supervised Classification Performance
On the unseen 40-customer test partition:
- **Accuracy**: **72.5%**
- **Precision**: **0.737**
- **Recall**: **0.700**
- **F1-Score**: **0.718**
- **ROC-AUC**: **0.680**

**Coefficient Interpretation**:
- `Age` ($\beta = -0.5911$, Odds Ratio $\approx 0.55$): Age is the most decisive demographic factor. Each standard-deviation increase in age reduces the odds of high mall spending by nearly half.
- `Annual Income` ($\beta = -0.0120$): Confirms that earning power alone is weakly correlated with mall spending frequency.

---

## 6. Web Application Architecture
The web application is built with Flask and Vanilla CSS:
- **`GET /`**: Home landing page with project mission, metrics, and segment overviews.
- **`GET /dashboard`**: Executive dashboard with 4 metric cards, interactive Chart.js scatter plot (with hover tooltips and centroid markers), demographic charts, and high-resolution Matplotlib chart toggles.
- **`GET /customers`**: Real-time filtering by cluster, search by Customer ID, customer records table, and segment descriptions.
- **`GET/POST /predict`**: Interactive prediction form validating numerical bounds, scaling features, predicting cluster ID via K-Means, and scoring spend propensity via Logistic Regression.
- **`GET /api/customer-data`**: REST endpoint serving clustered customer coordinates for client-side Chart.js rendering.

---

## 7. Limitations & Future Scope
1. **Static Cross-Sectional Data**: The dataset provides an aggregate snapshot without temporal timestamps or item-level transaction SKU data.
2. **Absence of Channel Attributes**: Physical mall foot traffic is represented, but omnichannel and e-commerce behaviors are absent.
3. **Future Enhancements**:
   - Integrate **RFM (Recency, Frequency, Monetary)** customer lifetime value modeling.
   - Implement **Density-Based Clustering (DBSCAN)** or **Gaussian Mixture Models (GMM)** for soft probabilistic cluster memberships.

---

## 8. Conclusion
The Customer Segmentation System successfully bridges machine learning theory and practical retail operations. By combining unsupervised K-Means clustering with supervised Logistic Regression propensity scoring, the system empowers retail managers to replace guesswork with personalized, data-driven marketing strategies. All artifacts, models, notebooks, and web interfaces have been tested and verified for immediate deployment.
