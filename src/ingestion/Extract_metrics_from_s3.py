import boto3
import tarfile
import io
import json

# --- Configuration ---
BUCKET = "ml-prod-pipeline"
REGION = "ap-south-1"

# The path to the tarball SageMaker created
# Replace [JOB_NAME] with your actual job name from logs
MODEL_TAR_KEY = "s3://ml-prod-pipeline/models/churn/xgboost/version=1/sagemaker-scikit-learn-2026-02-02-10-56-37-905/output/model.tar.gz"

# The new location for the extracted metrics
DESTINATION_METRICS_KEY = "s3://ml-prod-pipeline/models/churn/xgboost/version=1/sagemaker-scikit-learn-2026-02-02-10-56-37-905/output/metrics/metrics.json"


def extract_and_upload_metrics():
    s3 = boto3.client('s3', region_name=REGION)

    print(f"Downloading {MODEL_TAR_KEY}...")

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
                print("Found metrics.json inside tarball.")

                # 3. Upload the content back to the new S3 location
                s3.put_object(
                    Bucket=BUCKET,
                    Key=DESTINATION_METRICS_KEY,
                    Body=metrics_content,
                    ContentType='application/json'
                )
                print(f"🚀 Successfully uploaded to: s3://{BUCKET}/{DESTINATION_METRICS_KEY}")
            else:
                print("❌ metrics.json not found in the tarball!")
        except KeyError:
            print("❌ Could not find metrics.json in the archive. Check your train.py save logic.")


if __name__ == "__main__":
    extract_and_upload_metrics()