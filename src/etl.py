"""
etl.py
------
Extract, Transform, Load pipeline for Customer Churn Prediction.
- Extracts raw data from AWS S3 or local CSV
- Cleans and validates 500K+ records in chunks
- Loads processed data back to S3 and local disk
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config import (
    RAW_DIR, PROCESSED_DIR, CHUNK_SIZE,
    TARGET_COLUMN, CUSTOMER_ID_COL,
    S3_BUCKET, S3_RAW_PREFIX, S3_PROCESSED_PREFIX
)
from aws.s3_utils import download_from_s3, upload_to_s3

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────
def extract(source='s3', local_path=None):
    """
    Extract raw data from S3 or local file.

    Args:
        source (str): 's3' or 'local'
        local_path (str): path if source='local'

    Returns:
        pd.DataFrame: raw data
    """
    logger.info(f"Extracting data from: {source}")
    os.makedirs(RAW_DIR, exist_ok=True)

    if source == 's3':
        file_key = f"{S3_RAW_PREFIX}churn_data.csv"
        local_file = os.path.join(RAW_DIR, 'churn_data.csv')
        download_from_s3(S3_BUCKET, file_key, local_file)
        local_path = local_file

    chunks = []
    total_rows = 0

    # Read in chunks to handle 500K+ records efficiently
    for chunk in pd.read_csv(local_path, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info(f"  Loaded {total_rows:,} rows so far...")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"✅ Extraction complete: {len(df):,} total rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
def transform(df):
    """
    Clean and validate raw data.

    Steps:
    1. Remove duplicates
    2. Handle missing values
    3. Fix data types
    4. Encode target variable
    5. Validate business rules

    Args:
        df (pd.DataFrame): raw dataframe

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    logger.info("Starting data transformation...")
    original_shape = df.shape

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[CUSTOMER_ID_COL])
    logger.info(f"  Removed {before - len(df):,} duplicate customer IDs")

    # 2. Fix TotalCharges (sometimes has spaces instead of numbers)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        null_count = df['TotalCharges'].isna().sum()
        if null_count > 0:
            # Fill with MonthlyCharges * tenure (business logic)
            df['TotalCharges'] = df['TotalCharges'].fillna(
                df['MonthlyCharges'] * df['tenure']
            )
            logger.info(f"  Filled {null_count:,} missing TotalCharges values")

    # 3. Handle missing values in other columns
    for col in df.select_dtypes(include='object').columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna('Unknown')
            logger.info(f"  Filled {null_count:,} missing values in '{col}'")

    for col in df.select_dtypes(include='number').columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"  Filled {null_count:,} missing values in '{col}' with median")

    # 4. Encode target variable
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].dtype == object:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
        churn_rate = df[TARGET_COLUMN].mean() * 100
        logger.info(f"  Churn rate in dataset: {churn_rate:.2f}%")

    # 5. Fix SeniorCitizen column (sometimes 0/1, sometimes Yes/No)
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype == object:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    # 6. Validate: tenure must be >= 0
    if 'tenure' in df.columns:
        invalid = (df['tenure'] < 0).sum()
        if invalid > 0:
            df = df[df['tenure'] >= 0]
            logger.warning(f"  Removed {invalid} rows with negative tenure")

    # 7. Validate: MonthlyCharges must be > 0
    if 'MonthlyCharges' in df.columns:
        invalid = (df['MonthlyCharges'] <= 0).sum()
        if invalid > 0:
            df = df[df['MonthlyCharges'] > 0]
            logger.warning(f"  Removed {invalid} rows with zero/negative MonthlyCharges")

    logger.info(f"✅ Transform complete: {original_shape} → {df.shape}")
    return df


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load(df, destination='both'):
    """
    Save processed data to local disk and/or S3.

    Args:
        df (pd.DataFrame): processed dataframe
        destination (str): 'local', 's3', or 'both'
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename    = f'churn_processed_{timestamp}.csv'
    local_path  = os.path.join(PROCESSED_DIR, filename)
    latest_path = os.path.join(PROCESSED_DIR, 'churn_latest.csv')

    if destination in ('local', 'both'):
        df.to_csv(local_path, index=False)
        df.to_csv(latest_path, index=False)   # Always keep a 'latest' copy
        logger.info(f"✅ Saved locally: {local_path}")
        logger.info(f"✅ Saved as latest: {latest_path}")

    if destination in ('s3', 'both'):
        s3_key = f"{S3_PROCESSED_PREFIX}{filename}"
        upload_to_s3(local_path, S3_BUCKET, s3_key)
        logger.info(f"✅ Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")

    return local_path


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────
def etl_summary(df):
    """Print a concise summary of the processed dataset."""
    print("\n" + "=" * 50)
    print("  ETL SUMMARY")
    print("=" * 50)
    print(f"  Total Records  : {len(df):,}")
    print(f"  Features       : {df.shape[1]}")
    print(f"  Churn Rate     : {df[TARGET_COLUMN].mean()*100:.2f}%")
    print(f"  Churned        : {df[TARGET_COLUMN].sum():,}")
    print(f"  Not Churned    : {(df[TARGET_COLUMN]==0).sum():,}")
    print(f"  Missing Values : {df.isna().sum().sum()}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# MAIN ETL RUNNER
# ─────────────────────────────────────────────
def run_etl(source='local', local_path=None, destination='both'):
    """
    Full ETL pipeline runner.

    Args:
        source (str): 'local' or 's3'
        local_path (str): path to CSV if source='local'
        destination (str): where to save output

    Returns:
        str: path to processed file
    """
    logger.info("🚀 Starting ETL pipeline...")
    start = datetime.now()

    df         = extract(source=source, local_path=local_path)
    df_clean   = transform(df)
    output     = load(df_clean, destination=destination)

    etl_summary(df_clean)

    elapsed = (datetime.now() - start).seconds
    logger.info(f"✅ ETL complete in {elapsed}s → {output}")
    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ETL pipeline')
    parser.add_argument('--source',     default='local', choices=['local', 's3'])
    parser.add_argument('--input',      default='data/sample/sample_data.csv')
    parser.add_argument('--destination',default='local', choices=['local', 's3', 'both'])
    args = parser.parse_args()

    run_etl(source=args.source, local_path=args.input, destination=args.destination)
