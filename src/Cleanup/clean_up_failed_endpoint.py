import boto3
from src.logger import setup_logger

logger = setup_logger("cleanup_resources")

REGION = "ap-south-1"
ENDPOINT_NAME = "churn-xgboost-prod"
# Usually, the config and model names match or contain the endpoint name
CONFIG_NAME = ENDPOINT_NAME
MODEL_NAME = ENDPOINT_NAME

sm_client = boto3.client("sagemaker", region_name=REGION)


def cleanup():
    # 1. Delete Endpoint (This stops the billing)
    try:
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info(f"🗑️ Deleting Endpoint: {ENDPOINT_NAME}...")

        # Wait for the endpoint to actually disappear
        waiter = sm_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=ENDPOINT_NAME)
        logger.info("✅ Endpoint deleted successfully (Instance charges stopped).")
    except Exception as e:
        logger.warning(f"Note: Could not delete endpoint (it may not exist): {e}")

    # 2. Delete Endpoint Configuration
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=CONFIG_NAME)
        logger.info(f"🗑️ Deleting Config: {CONFIG_NAME}")
    except Exception as e:
        logger.warning(f"Note: Could not delete config: {e}")

    # 3. Delete the Model metadata
    try:
        sm_client.delete_model(ModelName=MODEL_NAME)
        logger.info(f"🗑️ Deleting Model: {MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Note: Could not delete model: {e}")


if __name__ == "__main__":
    cleanup()
    logger.info("✨ Cleanup complete. No further charges will be incurred for these resources.")