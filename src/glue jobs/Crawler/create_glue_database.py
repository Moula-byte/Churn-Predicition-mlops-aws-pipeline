import boto3
from botocore.exceptions import ClientError
from src.logger import setup_logger

logger=setup_logger("create_glue_database")


glue = boto3.client("glue", region_name="ap-south-1")

DATABASE_NAME = "churn_ml_db"

try:
    logger.info("Starting Glue Database creation")
    glue.create_database(
        DatabaseInput={
            "Name": DATABASE_NAME,
            "Description": "Glue database for churn ML pipeline"
        }
    )
    logger.info(f"Database '{DATABASE_NAME}' created")
except glue.exceptions.AlreadyExistsException:
    logger.error(f"ℹDatabase '{DATABASE_NAME}' already exists")
