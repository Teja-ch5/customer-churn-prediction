"""
aws/s3_utils.py
---------------
AWS S3 utility functions for uploading and downloading
data, models, and results.
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config import AWS_REGION

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def get_s3_client():
    """Create and return an S3 client."""
    return boto3.client('s3', region_name=AWS_REGION)


# ─────────────────────────────────────────────
# UPLOAD TO S3
# ─────────────────────────────────────────────
def upload_to_s3(local_path, bucket, s3_key):
    """
    Upload a local file to S3.

    Args:
        local_path (str): local file path
        bucket (str): S3 bucket name
        s3_key (str): S3 object key (path inside bucket)

    Returns:
        bool: True if successful
    """
    try:
        s3 = get_s3_client()
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB

        logger.info(f"Uploading {local_path} ({file_size:.2f} MB) → s3://{bucket}/{s3_key}")
        s3.upload_file(local_path, bucket, s3_key)
        logger.info(f"✅ Upload successful: s3://{bucket}/{s3_key}")
        return True

    except FileNotFoundError:
        logger.error(f"Local file not found: {local_path}")
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found. Run: aws configure")
        return False
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        return False


# ─────────────────────────────────────────────
# DOWNLOAD FROM S3
# ─────────────────────────────────────────────
def download_from_s3(bucket, s3_key, local_path):
    """
    Download a file from S3 to local disk.

    Args:
        bucket (str): S3 bucket name
        s3_key (str): S3 object key
        local_path (str): where to save locally

    Returns:
        bool: True if successful
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3 = get_s3_client()

        logger.info(f"Downloading s3://{bucket}/{s3_key} → {local_path}")
        s3.download_file(bucket, s3_key, local_path)
        logger.info(f"✅ Download successful: {local_path}")
        return True

    except NoCredentialsError:
        logger.error("AWS credentials not found. Run: aws configure")
        return False
    except ClientError as e:
        logger.error(f"S3 download failed: {e}")
        return False


# ─────────────────────────────────────────────
# LIST S3 OBJECTS
# ─────────────────────────────────────────────
def list_s3_objects(bucket, prefix=''):
    """List all objects in an S3 bucket with optional prefix."""
    try:
        s3       = get_s3_client()
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            logger.info(f"No objects found in s3://{bucket}/{prefix}")
            return []

        objects = [obj['Key'] for obj in response['Contents']]
        logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
        return objects

    except ClientError as e:
        logger.error(f"S3 list failed: {e}")
        return []


# ─────────────────────────────────────────────
# DELETE S3 OBJECT
# ─────────────────────────────────────────────
def delete_s3_object(bucket, s3_key):
    """Delete a single object from S3."""
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=bucket, Key=s3_key)
        logger.info(f"Deleted: s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"S3 delete failed: {e}")
        return False


# ─────────────────────────────────────────────
# UPLOAD ENTIRE FOLDER TO S3
# ─────────────────────────────────────────────
def upload_folder_to_s3(local_folder, bucket, s3_prefix):
    """
    Recursively upload a local folder to S3.

    Args:
        local_folder (str): local directory path
        bucket (str): S3 bucket name
        s3_prefix (str): S3 prefix (folder path in bucket)
    """
    for root, _, files in os.walk(local_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative   = os.path.relpath(local_path, local_folder)
            s3_key     = os.path.join(s3_prefix, relative).replace("\\", "/")
            upload_to_s3(local_path, bucket, s3_key)
