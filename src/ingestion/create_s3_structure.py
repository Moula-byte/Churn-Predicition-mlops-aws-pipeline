import boto3
from src.logger import setup_logger

logger=setup_logger("create s3 bucket folder Logger")

BUCKET_NAME = "ml-prod-pipeline"
REGION = "ap-south-1"

s3 = boto3.client("s3", region_name=REGION)

# Create bucket
try:
    logger.info("S3 Bucket Creating Started")
    s3.create_bucket(
        Bucket=BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": REGION}
    )

    logger.info(f"{BUCKET_NAME} Bucket created Sucessfully")
except s3.exceptions.BucketAlreadyOwnedByYou:
    logger.Error(f"Bucket Creation Failed due to BucketAlreadyOwnedByYou")

folders = [
    # RAW
    "raw/customers/",
    "raw/transactions/",
    "raw/usage/",
    "raw/support/",
    "raw/labels/",

    # QUARANTINE
    "quarantine/schema_mismatch/",
    "quarantine/null_threshold/",
    "quarantine/range_violation/",
    "quarantine/duplicates/",

    # PROCESSED
    "processed/base_table/",
    "processed/train/",
    "processed/validation/",
    "processed/test/",

    # FEATURES
    "features/offline/",
    "features/online/",

    # MODELS
    "models/artifacts/churn/",
    "models/registry/",

    # PREDICTIONS
    "predictions/batch/",
    "predictions/realtime/",

    # MONITORING
    "monitoring/data_drift/",
    "monitoring/model_quality/",
    "monitoring/bias/",

    # LOGS
    "logs/ingestion/",
    "logs/glue/",
    "logs/training/",
    "logs/endpoints/",
    "logs/stepfunctions/",
]

for folder in folders:
    s3.put_object(Bucket=BUCKET_NAME, Key=folder)
    logger.info(f"{folder} is created in {BUCKET_NAME} Bucket Sucessfully")

logger.info(" S3 production folder structure created")
