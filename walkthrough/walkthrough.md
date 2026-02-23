# E-Commerce Analytics Project - Complete Summary

## Project Overview

**Objective:** Perform end-to-end data analysis on the Olist E-Commerce dataset to derive actionable business insights using explainable methods suitable for non-technical stakeholders.

**Dataset:** Brazilian E-Commerce data from Olist marketplace
- **Time Period:** 2016-2018
- **Size:** ~100,000 orders, 113,000+ order items, 96,000+ customers

**Technology Stack:**
- **Language:** Python 3.x
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Approach:** Clean EDA, business logic, explainable models

---

## Phase 1: Data Loading & Validation

**Script:** `01_data_loading_validation.py`

### Datasets Loaded
| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| `olist_orders_dataset.csv` | 99,441 | 8 | Order-level data |
| `olist_order_items_dataset.csv` | 112,650 | 7 | Product items per order |
| `olist_customers_dataset.csv` | 99,441 | 5 | Customer information |

### Methods Used
- **pandas.read_csv()** - Data loading
- **DataFrame.shape** - Row/column counts
- **DataFrame.isnull().sum()** - Missing value detection
- **DataFrame.nunique()** - Primary key validation

### Key Findings
| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| `order_approved_at` | 160 | 0.16% |
| `order_delivered_carrier_date` | 1,783 | 1.79% |
| `order_delivered_customer_date` | 2,965 | 2.98% |

**Business Interpretation:** Missing delivery dates represent orders in transit, cancelled, or with data entry issues.

---

## Phase 2: Data Joining

**Script:** `02_data_joining.py`

### Join Strategy
```
Orders ──LEFT JOIN──> Customers (on customer_id)
   │
   └──LEFT JOIN──> Order Items (on order_id)
```

### Methods Used
- **pandas.merge()** with `how='left'` - Preserve all orders
- **validate='one_to_one'** - Ensure join integrity
- **groupby().size()** - Composite key validation

### Join Results
| Join | Before | After | Status |
|------|--------|-------|--------|
| Orders + Customers | 99,441 | 99,441 | ✅ Perfect 1:1 |
| Result + Order Items | 99,441 | 113,425 | ✅ Expected expansion |

**Why LEFT JOIN?** Preserves complete order history while enriching with customer/item details. Inner join would lose incomplete orders.

### Output
- `merged_ecommerce_data.csv` - 113,425 rows × 18 columns

---

## Phase 3: Order Funnel Analysis

**Script:** `03_order_funnel_analysis.py`

### Funnel Stage Definitions
| Stage | Statuses Included | Description |
|-------|-------------------|-------------|
| 1. Order Placed | created, approved, invoiced, processing, shipped, delivered | Order entered system |
| 2. Payment Approved | approved, invoiced, processing, shipped, delivered | Payment successful |
| 3. Shipped | shipped, delivered | Handed to carrier |
| 4. Delivered | delivered | Received by customer |

**Excluded:** `canceled` (625), `unavailable` (609) — Analyzed separately

### Methods Used
- **DataFrame.isin()** - Status classification
- **astype(int)** - Binary indicator creation
- **value_counts()** - Distribution analysis

### Funnel Metrics
| Stage | Orders | Conversion Rate |
|-------|--------|-----------------|
| Order Placed | 98,207 | 100% |
| Payment Approved | 98,202 | 99.99% |
| Shipped | 97,585 | 99.37% |
| Delivered | 96,478 | 98.87% |

**Overall Success Rate:** 98.24%

### Output
- `merged_ecommerce_with_funnel.csv` - Added 5 binary indicator columns

---

## Phase 4: Funnel Metrics & Visualization

**Script:** `04_funnel_metrics_visualization.py`

### Methods Used
- **nunique()** - COUNT DISTINCT order_id
- **matplotlib.pyplot** - Bar charts, waterfall charts
- **Custom drop-off calculations** - Stage-to-stage analysis

### Key Metrics
| Transition | Drop-off | Drop-off % |
|------------|----------|------------|
| Placed → Approved | 5 | 0.01% |
| Approved → Shipped | 617 | 0.63% |
| **Shipped → Delivered** | **1,107** | **1.13%** ⚠️ |

**Highest Drop-off:** Shipped → Delivered

### Business Reasons for Delivery Failures
1. Customer unavailable
2. Incorrect address
3. Package lost/damaged
4. Returned to sender

### Visualizations Created
- `funnel_visualization.png` - Bar charts
- `funnel_dropoff_waterfall.png` - Waterfall chart

---

## Phase 5: Geographic Funnel Analysis

**Script:** `05_geographic_funnel_analysis.py`

### Methods Used
- **groupby('customer_state')** - State-level aggregation
- **Calculated metrics:** Delivery rate, failure rate
- **pd.cut()** - Performance tier classification

### State Performance Rankings
**Top 5 States (500+ orders):**
| Rank | State | Delivery Rate |
|------|-------|---------------|
| 1 | MS | 99.01% |
| 2 | ES | 98.86% |
| 3 | PR | 98.82% |
| 4 | MG | 98.76% |
| 5 | RS | 98.69% |

**Bottom 5 States (500+ orders):**
| Rank | State | Delivery Rate |
|------|-------|---------------|
| 1 | CE | 96.67% ⚠️ |
| 2 | PE | 96.96% |
| 3 | RJ | 97.27% (high volume!) |
| 4 | PB | 97.36% |
| 5 | BA | 97.37% |

### Geographic Insights
- Southeast (SP, RJ, MG): Best infrastructure
- Northeast (CE, PE, BA): Higher failure rates
- Remote states: Higher freight, lower delivery success

### Visualizations
- `geographic_funnel_analysis.png` - 4 charts

---

## Phase 6: Revenue Analysis

**Script:** `06_revenue_analysis.py`

### Methods Used
- **DataFrame.sum()** - Total revenue calculation
- **groupby().agg()** - Multi-metric aggregation
- **DataFrame.quantile()** - Percentile analysis

### Revenue Metrics
| Metric | Value |
|--------|-------|
| **Total Revenue** | $15,843,553.24 |
| Product Revenue | $13,591,643.70 (85.8%) |
| Freight Revenue | $2,251,909.54 (14.2%) |
| **Average Order Value (AOV)** | $159.33 |
| Median Order Value | $104.56 |

### Revenue by State (Top 5)
| State | Revenue | Share | AOV |
|-------|---------|-------|-----|
| SP | $5,921,678 | 37.4% | $141.85 |
| RJ | $2,129,682 | 13.4% | $165.71 |
| MG | $1,856,161 | 11.7% | $159.53 |
| RS | $885,827 | 5.6% | $162.06 |
| PR | $800,935 | 5.1% | $158.76 |

**Key Finding:** Top 3 states = 62.5% of revenue (concentration risk)

### AOV Analysis
- **Highest AOV:** PB $263.04 (remote, high freight)
- **Lowest AOV:** SP $141.85 (high volume, competitive)
- **Gap:** $121.19 (85.4% difference)

### Visualizations
- `revenue_analysis_charts.png` - 4 charts

---

## Phase 7: Customer Repeat Analysis

**Script:** `07_customer_repeat_analysis.py`

### Key Concept: customer_id vs customer_unique_id
| Identifier | Description | Count |
|------------|-------------|-------|
| `customer_id` | Per-order ID (like receipt number) | 99,441 |
| `customer_unique_id` | Per-person ID (like loyalty card) | 96,096 |

**Critical:** Must use `customer_unique_id` to track repeat behavior!

### Methods Used
- **groupby('customer_unique_id')** - Customer-level aggregation
- **apply(lambda)** - Customer type classification
- **LabelEncoder** - Categorical encoding

### Customer Breakdown
| Customer Type | Count | Percentage |
|---------------|-------|------------|
| New (1 order) | 93,099 | 96.88% |
| Repeat (2+ orders) | 2,997 | **3.12%** |

### Revenue Contribution
| Type | Customers | Revenue | Avg LTV |
|------|-----------|---------|---------|
| New | 93,099 | $14.92M (94.2%) | $160.28 |
| Repeat | 2,997 | $0.92M (5.8%) | **$307.66** |

**Key Insight:** Repeat customers are worth **1.9x more** than new customers!

### Recommendations
1. Invest in retention marketing
2. Implement loyalty programs
3. Target: Increase repeat rate from 3% to 5%+

### Visualizations
- `customer_repeat_analysis_charts.png` - 4 charts

---

## Phase 8: Delivery Prediction Model

**Script:** `08_delivery_prediction_model.py`

### Machine Learning Approach
**Models Used:**
1. **Logistic Regression** - Linear, interpretable
2. **Decision Tree** - Non-linear, rule-based

**Why these models?**
- Simple, explainable to stakeholders
- No black-box neural networks
- Feature importance easily extracted

### Feature Engineering
| Feature | Type | Description |
|---------|------|-------------|
| `price` | Numeric | Order total |
| `freight_value` | Numeric | Shipping cost |
| `state_encoded` | Encoded | Customer state |
| `purchase_hour` | Numeric | Hour of order |
| `purchase_dayofweek` | Numeric | Day of week |
| `purchase_month` | Numeric | Month |

### Methods Used
- **train_test_split()** - 80/20 split, stratified
- **StandardScaler()** - Feature normalization
- **LogisticRegression()** - sklearn implementation
- **DecisionTreeClassifier()** - max_depth=4 for interpretability
- **confusion_matrix()** - Evaluation metric

### Model Results
| Model | Accuracy | True Neg | True Pos |
|-------|----------|----------|----------|
| Logistic Regression | 97.02% | 0 | 19,296 |
| Decision Tree | 97.80% | 156 | 19,296 |

### Feature Importance
**Decision Tree:**
1. `price` - 99.5%
2. `freight_value` - 0.3%
3. `purchase_month` - 0.1%

**Logistic Regression Coefficients:**
| Feature | Coefficient | Effect |
|---------|-------------|--------|
| `freight_value` | +0.52 | ↑ Increases delivery |
| `price` | -0.10 | ↓ Decreases delivery |
| `state_encoded` | +0.08 | Location matters |

### Business Interpretation
- **97% accuracy = baseline** (most orders deliver anyway)
- Model struggles to predict individual failures
- Best used for **strategic insights**, not operational predictions
- Confirms geography as main driver of delivery issues

### Visualizations
- `delivery_prediction_charts.png` - 4 charts

---

## Project Deliverables

### Python Scripts Created
| # | Script | Purpose |
|---|--------|---------|
| 1 | `01_data_loading_validation.py` | Load & validate data |
| 2 | `02_data_joining.py` | Join tables |
| 3 | `03_order_funnel_analysis.py` | Funnel stages |
| 4 | `04_funnel_metrics_visualization.py` | Funnel metrics |
| 5 | `05_geographic_funnel_analysis.py` | State analysis |
| 6 | `06_revenue_analysis.py` | Revenue metrics |
| 7 | `07_customer_repeat_analysis.py` | Customer behavior |
| 8 | `08_delivery_prediction_model.py` | ML prediction |

### Data Files Created
| File | Rows | Purpose |
|------|------|---------|
| `merged_ecommerce_data.csv` | 113,425 | Base joined data |
| `merged_ecommerce_with_funnel.csv` | 113,425 | With funnel indicators |

### Report Files Created
| File | Content |
|------|---------|
| `validation_results.txt` | Data quality report |
| `joining_results.txt` | Join validation |
| `funnel_analysis_results.txt` | Funnel metrics |
| `funnel_metrics_results.txt` | Conversion rates |
| `geographic_funnel_results.txt` | State performance |
| `revenue_analysis_results.txt` | Revenue metrics |
| `customer_repeat_analysis_results.txt` | Customer metrics |
| `delivery_prediction_results.txt` | Model results |

### Visualizations Created
| File | Charts |
|------|--------|
| `funnel_visualization.png` | Funnel bar charts |
| `funnel_dropoff_waterfall.png` | Waterfall chart |
| `geographic_funnel_analysis.png` | 4 state charts |
| `revenue_analysis_charts.png` | 4 revenue charts |
| `customer_repeat_analysis_charts.png` | 4 customer charts |
| `delivery_prediction_charts.png` | 4 model charts |

---

## Key Business Recommendations

### 1. Delivery Operations
- Focus on problem states: CE, PE, RJ, BA
- Partner with reliable carriers in Northeast
- Improve address validation at checkout

### 2. Revenue Growth
- Diversify beyond SP (37% concentration risk)
- Upselling opportunity in high-volume/low-AOV states
- Premium products for high-AOV states (PB, PA, CE)

### 3. Customer Retention
- Current repeat rate (3.1%) is LOW
- Each repeat customer worth 1.9x more
- Invest in loyalty programs & retention marketing

### 4. Strategic Use of ML
- Use model insights for strategic planning
- Don't rely on predictions for individual orders
- Geography is the main driver of delivery issues

---

## Technologies & Methods Summary

| Category | Technologies/Methods |
|----------|---------------------|
| **Data Loading** | pandas.read_csv() |
| **Data Cleaning** | isnull(), fillna(), drop_duplicates() |
| **Data Joining** | merge() with LEFT JOIN, validate param |
| **Aggregation** | groupby(), agg(), value_counts() |
| **Feature Engineering** | LabelEncoder, datetime parsing, cut() |
| **Visualization** | matplotlib.pyplot, bar charts, pie charts, scatter plots |
| **Machine Learning** | LogisticRegression, DecisionTreeClassifier |
| **Model Evaluation** | train_test_split, accuracy_score, confusion_matrix |
| **Scaling** | StandardScaler |

---

*Project completed: January 20, 2026*
