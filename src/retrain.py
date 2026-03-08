"""
retrain.py
----------
Automated model retraining scheduler.
- Downloads latest data from S3
- Runs ETL → Preprocessing → Training → Evaluation
- Deploys new model to S3 if performance improves
- Designed to run weekly via cron on AWS EC2

Cron setup (runs every Sunday midnight):
    0 0 * * 0 python /home/ec2-user/churn/src/retrain.py >> /var/log/churn_retrain.log 2>&1
"""

import os
import joblib
import logging
from datetime import datetime

from config import (
    MODELS_DIR, RESULTS_DIR,
    MIN_RECORDS_RETRAIN,
    S3_BUCKET, S3_MODELS_PREFIX
)
from src.etl          import run_etl
from src.preprocessing import preprocess
from src.train        import run_training
from src.evaluate     import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# GET CURRENT PRODUCTION MODEL ACCURACY
# ─────────────────────────────────────────────
def get_current_accuracy():
    """Load metadata of the currently deployed model."""
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.joblib')
    if not os.path.exists(meta_path):
        logger.warning("No existing model metadata found. Will train from scratch.")
        return 0.0

    metadata = joblib.load(meta_path)
    acc = metadata.get('val_accuracy', 0.0)
    logger.info(f"Current production model accuracy: {acc:.4f}")
    return acc


# ─────────────────────────────────────────────
# CHECK IF RETRAINING IS NEEDED
# ─────────────────────────────────────────────
def should_retrain(new_data_path):
    """
    Decide if retraining should proceed.
    Rules:
    1. New data must have at least MIN_RECORDS_RETRAIN rows
    2. Always retrain on schedule (weekly)
    """
    import pandas as pd
    df = pd.read_csv(new_data_path)

    if len(df) < MIN_RECORDS_RETRAIN:
        logger.warning(
            f"Only {len(df):,} new records. "
            f"Minimum required: {MIN_RECORDS_RETRAIN:,}. Skipping retrain."
        )
        return False

    logger.info(f"New data has {len(df):,} records. Proceeding with retrain.")
    return True


# ─────────────────────────────────────────────
# DEPLOY NEW MODEL IF BETTER
# ─────────────────────────────────────────────
def deploy_if_better(new_accuracy, current_accuracy, threshold=0.005):
    """
    Deploy the new model to S3 if it performs better than current.

    Args:
        threshold (float): minimum improvement required (0.5%)
    """
    improvement = new_accuracy - current_accuracy

    if improvement >= threshold:
        logger.info(
            f"✅ New model is better! "
            f"{current_accuracy:.4f} → {new_accuracy:.4f} "
            f"(+{improvement:.4f}). Deploying..."
        )
        try:
            from aws.s3_utils import upload_to_s3
            prod_path = os.path.join(MODELS_DIR, 'production_model.joblib')
            meta_path = os.path.join(MODELS_DIR, 'model_metadata.joblib')
            upload_to_s3(prod_path, S3_BUCKET, f"{S3_MODELS_PREFIX}production_model.joblib")
            upload_to_s3(meta_path, S3_BUCKET, f"{S3_MODELS_PREFIX}model_metadata.joblib")
            logger.info("✅ New model deployed to S3 successfully.")
        except Exception as e:
            logger.error(f"S3 deployment failed: {e}")
    else:
        logger.info(
            f"⚠️ New model did not improve enough. "
            f"Current: {current_accuracy:.4f}, New: {new_accuracy:.4f}. "
            f"Keeping existing production model."
        )


# ─────────────────────────────────────────────
# SEND NOTIFICATION (optional)
# ─────────────────────────────────────────────
def send_notification(status, details):
    """
    Send retraining status notification.
    Extend this to use SNS, Slack, or email.
    """
    logger.info(f"📬 Notification: [{status}] {details}")
    # TODO: Integrate with AWS SNS or Slack webhook
    # import boto3
    # sns = boto3.client('sns', region_name='us-east-1')
    # sns.publish(TopicArn='arn:aws:sns:...', Message=details, Subject=f'Churn Model Retrain: {status}')


# ─────────────────────────────────────────────
# MAIN RETRAINING PIPELINE
# ─────────────────────────────────────────────
def run_retraining():
    """Full automated retraining pipeline."""
    start     = datetime.now()
    run_date  = start.strftime('%Y-%m-%d %H:%M')

    logger.info("=" * 55)
    logger.info(f"  🔄 AUTOMATED RETRAINING — {run_date}")
    logger.info("=" * 55)

    try:
        # Step 1: Get current model accuracy
        current_acc = get_current_accuracy()

        # Step 2: Extract latest data from S3
        logger.info("Step 1/4: Running ETL pipeline...")
        processed_path = run_etl(source='s3', destination='both')

        # Step 3: Check if retraining should proceed
        if not should_retrain(processed_path):
            send_notification('SKIPPED', 'Insufficient new data for retraining.')
            return

        # Step 4: Train new models
        logger.info("Step 2/4: Training models...")
        results, metadata = run_training(input_path=processed_path, upload_s3=False)
        new_acc = metadata['val_accuracy']

        # Step 5: Evaluate
        logger.info("Step 3/4: Evaluating new model...")
        all_metrics = evaluate(input_path=processed_path)

        # Step 6: Deploy if better
        logger.info("Step 4/4: Deployment decision...")
        deploy_if_better(new_acc, current_acc)

        elapsed = (datetime.now() - start).seconds
        summary = (
            f"Retrain complete in {elapsed}s. "
            f"New accuracy: {new_acc:.4f} | Previous: {current_acc:.4f}"
        )
        logger.info(f"✅ {summary}")
        send_notification('SUCCESS', summary)

    except Exception as e:
        error_msg = f"Retraining failed: {str(e)}"
        logger.error(error_msg)
        send_notification('FAILED', error_msg)
        raise


if __name__ == '__main__':
    run_retraining()
