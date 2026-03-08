# 🔄 Customer Churn Prediction — Machine Learning on AWS

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red.svg)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-yellow.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25-brightgreen.svg)
![Records](https://img.shields.io/badge/Dataset-500K%2B%20Records-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade **end-to-end Machine Learning pipeline** for predicting customer churn using Logistic Regression, Random Forest, and XGBoost on **500K+ records**, deployed on **AWS S3 & EC2** with automated ETL retraining.

---

## 📌 Project Highlights

- ✅ **87% accuracy** on held-out test set
- ✅ **500,000+ records** processed via automated ETL pipeline
- ✅ **3 ML models** compared: Logistic Regression, Random Forest, XGBoost
- ✅ **Deployed on AWS** (S3 for storage, EC2 for compute)
- ✅ **Automated retraining** pipeline with scheduled ETL jobs
- ✅ **Business insights** — top churn drivers identified and presented to stakeholders

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 config.py                    → All paths, hyperparameters, AWS config
│
├── 📁 data/
│   ├── raw/                        → Raw data (gitignored, stored on S3)
│   ├── processed/                  → Cleaned & feature-engineered data
│   └── sample/
│       └── sample_data.csv         → 1000-row sample for testing
│
├── 📁 src/
│   ├── etl.py                      → Extract, Transform, Load pipeline
│   ├── preprocessing.py            → Feature engineering & data cleaning
│   ├── train.py                    → Model training (LR, RF, XGBoost)
│   ├── evaluate.py                 → Metrics, plots, feature importance
│   ├── predict.py                  → Run inference on new data
│   └── retrain.py                  → Automated retraining scheduler
│
├── 📁 aws/
│   ├── s3_utils.py                 → Upload/download data from S3
│   └── ec2_deploy.sh               → EC2 setup & deployment script
│
├── 📁 notebooks/
│   └── churn_analysis.ipynb        → Full EDA + modeling notebook
│
└── 📁 results/
    └── model_comparison.md         → Model comparison & business insights
```

---

## 📊 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 81.2% | 79.4% | 76.8% | 78.1% | 0.84 |
| Random Forest | 85.6% | 84.1% | 82.3% | 83.2% | 0.91 |
| **XGBoost** ⭐ | **87.0%** | **86.2%** | **85.1%** | **85.6%** | **0.93** |

> **XGBoost** was selected as the final model for production deployment.

### Top Churn Drivers Identified
1. **Contract Type** — Month-to-month customers churn 3x more than annual
2. **Tenure** — Customers < 6 months have 65% churn probability
3. **Monthly Charges** — High charges (>$70/mo) correlate strongly with churn
4. **Tech Support** — Customers without tech support churn 2x more
5. **Internet Service** — Fiber optic users show higher churn rates

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| ML Models | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Cloud Storage | AWS S3 (boto3) |
| Cloud Compute | AWS EC2 |
| ETL Pipeline | Python + Cron/Scheduler |
| Model Serialization | Joblib |
| Statistics | SciPy, Statsmodels |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure AWS credentials
```bash
aws configure
# Enter: AWS Access Key, Secret Key, Region (us-east-1)
```

### 5. Update config
Edit `config.py` with your S3 bucket name and EC2 details.

---

## 🚀 Running the Pipeline

```bash
# Step 1: Run ETL (extract, clean, load data)
python src/etl.py

# Step 2: Preprocess & feature engineer
python src/preprocessing.py

# Step 3: Train all models
python src/train.py

# Step 4: Evaluate & compare models
python src/evaluate.py

# Step 5: Predict on new data
python src/predict.py --input data/sample/sample_data.csv

# Step 6: Deploy retraining scheduler on EC2
python src/retrain.py
```

---

## ☁️ AWS Deployment

### S3 — Data Storage
```
s3://your-bucket/
├── raw-data/           → incoming customer data
├── processed-data/     → cleaned features
├── models/             → serialized .joblib model files
└── results/            → prediction outputs & reports
```

### EC2 — Model Training & Retraining
```bash
# Setup EC2 instance (run once)
bash aws/ec2_deploy.sh

# Automated retraining runs every Sunday at midnight via cron
0 0 * * 0 python /home/ec2-user/churn/src/retrain.py
```

---

## 📈 Business Recommendations

Based on model outputs and statistical analysis:

| Segment | Churn Risk | Recommended Action |
|---------|-----------|-------------------|
| Month-to-month, <6 months tenure | 🔴 Very High | Offer annual contract discount |
| Monthly charges > $70, no support | 🔴 High | Proactive tech support outreach |
| Fiber optic, no security add-ons | 🟠 Medium | Bundle security + streaming offers |
| Long tenure (>24 months) | 🟢 Low | Loyalty rewards program |

---

## 📄 License

MIT License — feel free to use and adapt for your own projects.
