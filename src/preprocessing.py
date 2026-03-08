"""
preprocessing.py
----------------
Feature engineering and data preprocessing for Customer Churn Prediction.
- Label encoding & one-hot encoding
- Feature scaling
- Train/validation/test split
- Class imbalance handling (SMOTE)
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

from config import (
    PROCESSED_DIR, MODELS_DIR,
    TARGET_COLUMN, CUSTOMER_ID_COL,
    CATEGORICAL_COLS, NUMERICAL_COLS, DROP_COLUMNS,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    """
    Create new features from existing ones.

    New features:
    - tenure_group        : bucketed tenure (new/mid/long-term)
    - charges_per_month   : TotalCharges / tenure
    - has_multiple_services: count of add-on services
    - is_high_value       : MonthlyCharges > 70
    """
    df = df.copy()

    # Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 6, 12, 24, 48, 72],
        labels=['0-6mo', '7-12mo', '13-24mo', '25-48mo', '49-72mo']
    ).astype(str)

    # Charges per month of tenure
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Count of add-on services
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    present = [c for c in service_cols if c in df.columns]
    if present:
        df['num_services'] = (df[present] == 'Yes').sum(axis=1)

    # High value customer flag
    df['is_high_value'] = (df['MonthlyCharges'] > 70).astype(int)

    # Month-to-month contract flag
    if 'Contract' in df.columns:
        df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)

    # New customer flag
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

    logger.info(f"  Feature engineering: {df.shape[1]} total features")
    return df


# ─────────────────────────────────────────────
# ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
def encode_features(df, fit=True, encoders=None):
    """
    Encode categorical columns using one-hot encoding.

    Args:
        df (pd.DataFrame): input data
        fit (bool): True for training, False for inference
        encoders (dict): fitted encoders (for inference)

    Returns:
        pd.DataFrame, dict of encoders
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    # Binary Yes/No columns → 0/1
    binary_cols = [
        c for c in df.columns
        if df[c].dtype == object and set(df[c].dropna().unique()).issubset({'Yes', 'No', 'Unknown'})
    ]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Unknown': 0})

    # One-hot encode remaining categorical columns
    cat_cols = [
        c for c in CATEGORICAL_COLS
        if c in df.columns and df[c].dtype == object
    ]

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logger.info(f"  One-hot encoded {len(cat_cols)} categorical columns")

    return df, encoders


# ─────────────────────────────────────────────
# SCALE NUMERICAL FEATURES
# ─────────────────────────────────────────────
def scale_features(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data, transform all splits.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    num_cols = [c for c in NUMERICAL_COLS + ['charges_per_tenure'] if c in X_train.columns]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    logger.info(f"  Scaled {len(num_cols)} numerical columns")
    return X_train, X_val, X_test, scaler


# ─────────────────────────────────────────────
# HANDLE CLASS IMBALANCE (Oversampling)
# ─────────────────────────────────────────────
def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance via oversampling minority class.
    Typical churn datasets: ~20-25% churn (minority).

    Returns:
        X_balanced, y_balanced
    """
    df_train = pd.concat([X_train, y_train], axis=1)
    majority = df_train[df_train[TARGET_COLUMN] == 0]
    minority = df_train[df_train[TARGET_COLUMN] == 1]

    before_ratio = len(minority) / len(majority) * 100
    logger.info(f"  Class ratio before balancing: {before_ratio:.1f}% minority")

    # Oversample minority to match majority
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE
    )

    df_balanced  = pd.concat([majority, minority_upsampled])
    X_balanced   = df_balanced.drop(columns=[TARGET_COLUMN])
    y_balanced   = df_balanced[TARGET_COLUMN]

    logger.info(f"  After balancing: {len(X_balanced):,} training samples (50/50 split)")
    return X_balanced, y_balanced


# ─────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def preprocess(input_path=None, save=True):
    """
    Full preprocessing pipeline:
    1. Load cleaned data
    2. Engineer features
    3. Encode categoricals
    4. Split into train/val/test
    5. Scale numericals
    6. Handle class imbalance

    Returns:
        dict with X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load
    if input_path is None:
        input_path = os.path.join(PROCESSED_DIR, 'churn_latest.csv')

    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Feature engineering
    df = engineer_features(df)

    # Drop unnecessary columns
    drop = [c for c in DROP_COLUMNS + [CUSTOMER_ID_COL] if c in df.columns]
    df   = df.drop(columns=drop)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Encode
    X, encoders = encode_features(X)

    # Split: train / val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    logger.info(f"Split sizes → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Scale
    X_train, X_val, X_test, scaler = scale_features(X_train.copy(), X_val.copy(), X_test.copy())

    # Balance training set
    X_train, y_train = handle_imbalance(X_train, y_train)

    if save:
        joblib.dump(scaler,   os.path.join(MODELS_DIR, 'scaler.joblib'))
        joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.joblib'))
        joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, 'feature_names.joblib'))
        logger.info(f"✅ Scaler and encoders saved to {MODELS_DIR}")

    logger.info("✅ Preprocessing complete")
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val'  : X_val,   'y_val'  : y_val,
        'X_test' : X_test,  'y_test' : y_test,
        'scaler' : scaler,
        'feature_names': list(X_train.columns)
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, help='Path to processed CSV')
    args = parser.parse_args()

    data = preprocess(input_path=args.input)
    print(f"\nFeature count : {data['X_train'].shape[1]}")
    print(f"Train samples : {len(data['X_train']):,}")
    print(f"Val samples   : {len(data['X_val']):,}")
    print(f"Test samples  : {len(data['X_test']):,}")
