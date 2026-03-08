"""
config.py
---------
Central configuration for the Customer Churn Prediction pipeline.
Edit AWS credentials, paths, and hyperparameters here.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, 'data')
RAW_DIR         = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR   = os.path.join(DATA_DIR, 'processed')
SAMPLE_DIR      = os.path.join(DATA_DIR, 'sample')
RESULTS_DIR     = os.path.join(BASE_DIR, 'results')
MODELS_DIR      = os.path.join(BASE_DIR, 'models')

# ─────────────────────────────────────────────
# AWS Configuration
# ─────────────────────────────────────────────
AWS_REGION          = 'us-east-1'
S3_BUCKET           = 'your-churn-bucket'          # ← change this
S3_RAW_PREFIX       = 'raw-data/'
S3_PROCESSED_PREFIX = 'processed-data/'
S3_MODELS_PREFIX    = 'models/'
S3_RESULTS_PREFIX   = 'results/'

EC2_INSTANCE_ID     = 'i-xxxxxxxxxxxxxxxxx'        # ← change this
EC2_KEY_PATH        = '~/.ssh/your-key.pem'        # ← change this
EC2_USER            = 'ec2-user'

# ─────────────────────────────────────────────
# Data Configuration
# ─────────────────────────────────────────────
TARGET_COLUMN   = 'Churn'
CUSTOMER_ID_COL = 'customerID'
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1
RANDOM_STATE    = 42

# Features to drop before training
DROP_COLUMNS = [
    'customerID',
    'TotalCharges_raw'
]

# Categorical columns to encode
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Numerical columns to scale
NUMERICAL_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

# ─────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────
LR_PARAMS = {
    'C'           : 1.0,
    'max_iter'    : 1000,
    'random_state': RANDOM_STATE,
    'solver'      : 'lbfgs'
}

RF_PARAMS = {
    'n_estimators' : 200,
    'max_depth'    : 10,
    'min_samples_split': 5,
    'min_samples_leaf' : 2,
    'random_state' : RANDOM_STATE,
    'n_jobs'       : -1
}

XGB_PARAMS = {
    'n_estimators'      : 300,
    'max_depth'         : 6,
    'learning_rate'     : 0.05,
    'subsample'         : 0.8,
    'colsample_bytree'  : 0.8,
    'use_label_encoder' : False,
    'eval_metric'       : 'logloss',
    'random_state'      : RANDOM_STATE,
    'n_jobs'            : -1
}

# Best model to use in production
PRODUCTION_MODEL = 'xgboost'

# ─────────────────────────────────────────────
# ETL Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE          = 50_000        # rows per chunk for 500K+ data
RETRAIN_SCHEDULE    = 'weekly'      # 'daily' | 'weekly' | 'monthly'
MIN_RECORDS_RETRAIN = 10_000        # minimum new records to trigger retrain

# ─────────────────────────────────────────────
# Churn Risk Thresholds
# ─────────────────────────────────────────────
RISK_HIGH   = 0.70   # prob > 70% → High risk
RISK_MEDIUM = 0.40   # prob > 40% → Medium risk
                     # else       → Low risk
