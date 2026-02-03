import boto3
import tarfile
import io
import json
from src.logger import setup_logger

logger=setup_logger("Extract_metrics_from_s3")
# --- Configuration ---
BUCKET = "ml-prod-pipeline"
REGION = "ap-south-1"

# The path to the tarball SageMaker created
# Replace [JOB_NAME] with your actual job name from logs
MODEL_TAR_KEY = "models/churn/xgboost/version=1/sagemaker-scikit-learn-2026-02-02-10-56-37-905/output/model.tar.gz"
logger.info(F"Model Registry Path: {MODEL_TAR_KEY}")
# The new location for the extracted metrics
DESTINATION_METRICS_KEY = "models/churn/xgboost/version=1/sagemaker-scikit-learn-2026-02-02-10-56-37-905/output/metrics/metrics.json"
logger.info(f"destination Path is : {DESTINATION_METRICS_KEY}")


''' 
PREFIX = "models/churn/xgboost/version=1/"

s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)

if 'Contents' in response:
    print("Found these files in your bucket:")
    for obj in response['Contents']:
        print(f"{obj['Key']}")
else:
    print("No files found. Check your BUCKET name and PREFIX.")
'''


def extract_and_upload_metrics():
    s3 = boto3.client('s3', region_name=REGION)

    logger.info(f"Downloading {MODEL_TAR_KEY}...")

    # 1. Get the tarball from S3 into memory
    response = s3.get_object(Bucket=BUCKET, Key=MODEL_TAR_KEY)
    tar_bytes = io.BytesIO(response['Body'].read())

    # 2. Extract metrics.json from the tarball
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
        try:
            # Look for metrics.json inside the tar
            metrics_file = tar.extractfile("metrics.json")
            if metrics_file:
                metrics_content = metrics_file.read()
                logger.info("Found metrics.json inside tarball.")
                # 3. Upload the content back to the new S3 location
                s3.put_object(
                    Bucket=BUCKET,
                    Key=DESTINATION_METRICS_KEY,
                    Body=metrics_content,
                    ContentType='application/json'
                )
                logger.info(f"Successfully uploaded to: s3://{BUCKET}/{DESTINATION_METRICS_KEY}")
            else:
                logger.info("metrics.json not found in the tarball!")
        except KeyError:
            logger.error("Could not find metrics.json in the archive. Check your train.py save logic.")


if __name__ == "__main__":
    extract_and_upload_metrics()