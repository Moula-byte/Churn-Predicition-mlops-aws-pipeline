import boto3
from botocore.exceptions import NoCredentialsError
from src.logger import  setup_logger

logger= setup_logger('ingest_churnData')

def upload_file_to_s3(local_file_path, bucket_name, s3_object_key):
    """
    Uploads a file from a local path to an S3 bucket.

    :param local_file_path: The path to the file to upload (e.g., 'C:/path/to/your/file.txt' or './local_file.txt').
    :param bucket_name: The name of the S3 bucket.
    :param s3_object_key: The desired name and path for the object in S3 (e.g., 'folder/file.txt').
    """
    # Create an S3 client object (it will automatically use configured credentials)
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
        logger.info(f"Upload Successful: '{local_file_path}' copied to 's3://{bucket_name}/{s3_object_key}'")
        return True
    except FileNotFoundError:
        logger.Error(f"Error: The file '{local_file_path}' was not found")
        return False
    except NoCredentialsError:
        logger.Error("Error: AWS credentials not available or configured")
        return False
    except Exception as e:
        # Catch other potential errors like bucket not existing, permission issues, etc.
        logger.Error(f"An error occurred: {e}")
        return False

# --- Example Usage ---
# Replace with your specific details
LOCAL_FILE = 'D:/prctice/ML/AWS Churn Predicition MLops project/data/raw/ChurnData.csv'
S3_BUCKET = 'ml-prod-pipeline'
S3_KEY = 'processed/base_table/snapshot_date=2026-01-29/churn_base_20260129.csv'

upload_file_to_s3(LOCAL_FILE, S3_BUCKET, S3_KEY)
