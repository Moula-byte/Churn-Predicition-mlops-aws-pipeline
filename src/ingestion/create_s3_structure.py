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
    "raw/sensors/",
    "raw/maintenance_logs/",
    "raw/external/",

    "quarantine/",

    "processed/train/",
    "processed/validation/",
    "processed/test/",

    "features/offline/",
    "features/online/",

    "models/artifacts/",
    "models/registry/",

    "endpoints/logs/",

    "predictions/batch/",
    "predictions/realtime/",

    "monitoring/data_drift/",
    "monitoring/model_quality/",

    "logs/glue/",
    "logs/sagemaker/",
    "logs/stepfunctions/",
]

for folder in folders:
    s3.put_object(Bucket=BUCKET_NAME, Key=folder)
    logger.info(f"{folder} is created in {BUCKET_NAME} Bucket Sucessfully")

logger.info(" S3 production folder structure created")
