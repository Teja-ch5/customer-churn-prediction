"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost models
for customer churn prediction. Saves best model to disk and S3.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost               import XGBClassifier

from config import (
    MODELS_DIR, RESULTS_DIR,
    LR_PARAMS, RF_PARAMS, XGB_PARAMS,
    PRODUCTION_MODEL, RANDOM_STATE,
    S3_BUCKET, S3_MODELS_PREFIX
)
from src.preprocessing import preprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models():
    return {
        'logistic_regression': LogisticRegression(**LR_PARAMS),
        'random_forest'      : RandomForestClassifier(**RF_PARAMS),
        'xgboost'            : XGBClassifier(**XGB_PARAMS)
    }


# ─────────────────────────────────────────────
# TRAIN A SINGLE MODEL
# ─────────────────────────────────────────────
def train_model(name, model, X_train, y_train, X_val, y_val):
    """
    Train a single model and evaluate on validation set.

    Returns:
        trained model, val_accuracy
    """
    logger.info(f"Training {name}...")
    start = datetime.now()

    model.fit(X_train, y_train)

    # Cross-validation score on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    val_acc   = model.score(X_val, y_val)

    elapsed = (datetime.now() - start).seconds
    logger.info(
        f"  ✅ {name} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
        f"| Val Acc: {val_acc:.4f} | Time: {elapsed}s"
    )

    return model, val_acc


# ─────────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all(data):
    """
    Train all 3 models and return results.

    Args:
        data (dict): output from preprocess()

    Returns:
        dict: {model_name: (model, val_accuracy)}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']

    models  = get_models()
    results = {}

    print("\n" + "=" * 55)
    print("  TRAINING ALL MODELS")
    print("=" * 55)

    for name, model in models.items():
        trained_model, val_acc = train_model(name, model, X_train, y_train, X_val, y_val)
        results[name] = (trained_model, val_acc)

        # Save each model
        model_path = os.path.join(MODELS_DIR, f'{name}.joblib')
        joblib.dump(trained_model, model_path)
        logger.info(f"  Saved: {model_path}")

    # Print comparison table
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON (Validation Accuracy)")
    print("=" * 55)
    print(f"  {'Model':<25} {'Val Accuracy':>12}")
    print("-" * 40)
    for name, (_, acc) in sorted(results.items(), key=lambda x: -x[1][1]):
        marker = " ⭐ BEST" if name == max(results, key=lambda k: results[k][1]) else ""
        print(f"  {name:<25} {acc:>12.4f}{marker}")
    print("=" * 55 + "\n")

    return results


# ─────────────────────────────────────────────
# SAVE PRODUCTION MODEL
# ─────────────────────────────────────────────
def save_production_model(results, upload_to_s3=False):
    """
    Save the best model as the production model.

    Args:
        results (dict): model results from train_all()
        upload_to_s3 (bool): whether to upload to AWS S3
    """
    # Select best model by validation accuracy
    best_name  = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    best_acc   = results[best_name][1]

    prod_path  = os.path.join(MODELS_DIR, 'production_model.joblib')
    joblib.dump(best_model, prod_path)
    logger.info(f"✅ Production model saved: {best_name} (val_acc={best_acc:.4f})")

    # Save metadata
    metadata = {
        'model_name'    : best_name,
        'val_accuracy'  : best_acc,
        'trained_at'    : datetime.now().isoformat(),
        'all_results'   : {k: v[1] for k, v in results.items()}
    }
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.joblib')
    joblib.dump(metadata, meta_path)

    if upload_to_s3:
        try:
            from aws.s3_utils import upload_to_s3 as s3_upload
            s3_upload(prod_path, S3_BUCKET, f"{S3_MODELS_PREFIX}production_model.joblib")
            s3_upload(meta_path, S3_BUCKET, f"{S3_MODELS_PREFIX}model_metadata.joblib")
            logger.info(f"✅ Model uploaded to S3: s3://{S3_BUCKET}/{S3_MODELS_PREFIX}")
        except Exception as e:
            logger.warning(f"S3 upload skipped: {e}")

    return prod_path, metadata


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_training(input_path=None, upload_s3=False):
    """Full training pipeline."""
    logger.info("🚀 Starting model training pipeline...")

    # Preprocess
    data = preprocess(input_path=input_path)

    # Train all models
    results = train_all(data)

    # Save production model
    prod_path, metadata = save_production_model(results, upload_to_s3=upload_s3)

    logger.info("✅ Training pipeline complete!")
    logger.info(f"   Best model : {metadata['model_name']}")
    logger.info(f"   Val accuracy: {metadata['val_accuracy']:.4f}")

    return results, metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     default=None, help='Path to processed CSV')
    parser.add_argument('--upload-s3', action='store_true', help='Upload model to S3')
    args = parser.parse_args()

    run_training(input_path=args.input, upload_s3=args.upload_s3)
