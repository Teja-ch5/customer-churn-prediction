"""
predict.py
----------
Run churn predictions on new customer data.
Outputs churn probability + risk segment for each customer.

Usage:
    python src/predict.py --input data/new_customers.csv --output results/predictions.csv
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import logging

from config import MODELS_DIR, RESULTS_DIR, CUSTOMER_ID_COL, RISK_HIGH, RISK_MEDIUM
from src.preprocessing import engineer_features, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LOAD PRODUCTION MODEL & ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts():
    """Load trained model, scaler, and feature names."""
    prod_path    = os.path.join(MODELS_DIR, 'production_model.joblib')
    scaler_path  = os.path.join(MODELS_DIR, 'scaler.joblib')
    features_path = os.path.join(MODELS_DIR, 'feature_names.joblib')

    for path in [prod_path, scaler_path, features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}\nRun train.py first.")

    model         = joblib.load(prod_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    logger.info(f"Model loaded: {prod_path}")
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# ASSIGN RISK SEGMENT
# ─────────────────────────────────────────────
def assign_risk(prob):
    if prob >= RISK_HIGH:
        return 'HIGH'
    elif prob >= RISK_MEDIUM:
        return 'MEDIUM'
    return 'LOW'


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(input_path, output_path=None):
    """
    Run predictions on new customer data.

    Args:
        input_path (str): Path to CSV with new customers
        output_path (str): Where to save predictions CSV

    Returns:
        pd.DataFrame: predictions with churn probability and risk segment
    """
    model, scaler, feature_names = load_artifacts()

    # Load input data
    logger.info(f"Loading input data: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"  {len(df):,} customers to score")

    # Keep customer IDs if present
    customer_ids = df[CUSTOMER_ID_COL].copy() if CUSTOMER_ID_COL in df.columns else pd.Series(range(len(df)))

    # Preprocess
    df_feat = engineer_features(df.copy())

    if CUSTOMER_ID_COL in df_feat.columns:
        df_feat = df_feat.drop(columns=[CUSTOMER_ID_COL])

    df_feat, _ = encode_features(df_feat, fit=False)

    # Align features with training columns
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    df_feat = df_feat[feature_names]

    # Scale
    from config import NUMERICAL_COLS
    num_cols = [c for c in NUMERICAL_COLS + ['charges_per_tenure'] if c in df_feat.columns]
    df_feat[num_cols] = scaler.transform(df_feat[num_cols])

    # Predict
    proba   = model.predict_proba(df_feat)[:, 1]
    pred    = model.predict(df_feat)

    # Build results dataframe
    results = pd.DataFrame({
        CUSTOMER_ID_COL     : customer_ids.values,
        'churn_prediction'  : pred,
        'churn_probability' : proba.round(4),
        'risk_segment'      : [assign_risk(p) for p in proba],
        'recommendation'    : [get_recommendation(p) for p in proba]
    })

    # Summary
    print(f"\n{'='*55}")
    print(f"  CHURN PREDICTION SUMMARY")
    print(f"{'='*55}")
    print(f"  Total Customers  : {len(results):,}")
    print(f"  Predicted Churn  : {pred.sum():,}  ({pred.mean()*100:.1f}%)")
    risk_counts = results['risk_segment'].value_counts()
    print(f"  🔴 High Risk     : {risk_counts.get('HIGH', 0):,}")
    print(f"  🟠 Medium Risk   : {risk_counts.get('MEDIUM', 0):,}")
    print(f"  🟢 Low Risk      : {risk_counts.get('LOW', 0):,}")
    print(f"{'='*55}\n")

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"✅ Predictions saved: {output_path}")

    return results


# ─────────────────────────────────────────────
# BUSINESS RECOMMENDATIONS
# ─────────────────────────────────────────────
def get_recommendation(prob):
    if prob >= RISK_HIGH:
        return "Immediate outreach — offer annual contract discount or loyalty reward"
    elif prob >= RISK_MEDIUM:
        return "Proactive support call — bundle upgrade or tech support offer"
    return "Standard engagement — loyalty program inclusion"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Customer Churn Predictor')
    parser.add_argument('--input',  required=True, help='Path to input CSV')
    parser.add_argument('--output', default='results/predictions.csv', help='Output CSV path')
    args = parser.parse_args()

    results = predict(args.input, args.output)
    print(results.head(10).to_string(index=False))
