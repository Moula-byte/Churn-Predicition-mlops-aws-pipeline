import boto3
from src.logger import setup_logger

logger = setup_logger("approve_model")

REGION = "ap-south-1"
MODEL_PACKAGE_GROUP = "churn-xgboost-models"

sm = boto3.client("sagemaker", region_name=REGION)

def approve_latest_model():
    try:
        # 1. Dynamically fetch the latest model version created
        # We sort by CreationTime to ensure we get the newest iteration
        response = sm.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )

        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            logger.error(f"No model versions found in group: {MODEL_PACKAGE_GROUP}")
            return

        latest_arn = packages[0]["ModelPackageArn"]
        current_status = packages[0]["ModelApprovalStatus"]

        logger.info(f"Latest version found: {latest_arn} (Current Status: {current_status})")

        # 2. Only update if it's not already approved
        if current_status != "Approved":
            sm.update_model_package(
                ModelPackageArn=latest_arn,
                ModelApprovalStatus="Approved",
                ApprovalDescription="Dynamic approval via MLOps pipeline for latest version"
            )
            logger.info(f"✅ Successfully approved: {latest_arn}")
        else:
            logger.info("Model is already in 'Approved' status. No action taken.")

    except Exception as e:
        logger.error(f"❌ Error during dynamic approval: {str(e)}")

if __name__ == "__main__":
    approve_latest_model()