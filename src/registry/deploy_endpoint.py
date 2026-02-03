import boto3
import sagemaker
from sagemaker.model import ModelPackage
from src.logger import setup_logger

logger = setup_logger("deploy_endpoint")

# Configuration
REGION = "ap-south-1"
ROLE = "arn:aws:iam::998331943727:role/SageMakerExecutionRole-MLProd"
MODEL_PACKAGE_GROUP = "churn-xgboost-models"
ENDPOINT_NAME = "churn-xgboost-prod"

# Setup SageMaker Session
boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client("sagemaker")


def get_latest_approved_model_arn():
    """Dynamically retrieves the ARN of the latest approved model version."""
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )

        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            raise ValueError(f"No approved models found in group: {MODEL_PACKAGE_GROUP}")

        latest_arn = packages[0]["ModelPackageArn"]
        logger.info(f"Found latest approved model: {latest_arn}")
        return latest_arn
    except Exception as e:
        logger.error(f"Error fetching latest approved model: {e}")
        raise


def is_endpoint_running(name):
    """Checks if the endpoint already exists in the account."""
    try:
        sm_client.describe_endpoint(EndpointName=name)
        return True
    except sm_client.exceptions.ClientError:
        return False


def deploy():
    try:
        # 1. Get the ARN of the model version you just approved (e.g., Version 2)
        model_package_arn = get_latest_approved_model_arn()

        # 2. Create the Model object from the Registry ARN
        model = ModelPackage(
            role=ROLE,
            model_package_arn=model_package_arn,
            sagemaker_session=sagemaker_session
        )

        # 3. Dynamic Deployment Logic
        if is_endpoint_running(ENDPOINT_NAME):
            logger.info(f"Endpoint '{ENDPOINT_NAME}' exists. Starting Blue/Green update...")
            # update_endpoint=True performs a seamless transition to the new version
            model.deploy(
                initial_instance_count=1,
                instance_type="ml.m5.xlarge",
                endpoint_name=ENDPOINT_NAME,
                update_endpoint=True
            )
            logger.info(f"ndpoint updated successfully with version from {model_package_arn}")
        else:
            logger.info(f"Creating fresh endpoint: '{ENDPOINT_NAME}'")
            model.deploy(
                initial_instance_count=1,
                instance_type="ml.m5.xlarge",
                endpoint_name=ENDPOINT_NAME
            )
            logger.info(f"Initial deployment successful.")

    except Exception as e:
        logger.error(f" Deployment workflow failed: {e}")
        raise


if __name__ == "__main__":
    deploy()