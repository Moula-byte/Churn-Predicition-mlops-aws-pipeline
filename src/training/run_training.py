import sagemaker
import boto3
import os
from sagemaker.sklearn.estimator import SKLearn


from src.logger import setup_logger

logger = setup_logger("run_training")



REGION = "ap-south-1"
ROLE = "arn:aws:iam::998331943727:role/SageMakerExecutionRole-MLProd"

BUCKET = "ml-prod-pipeline"

TRAIN_PATH = f"s3://{BUCKET}/processed/model_splits/train/snapshot_date=2026-01-29/"
VAL_PATH   = f"s3://{BUCKET}/processed/model_splits/validation/snapshot_date=2026-01-29/"
MODEL_PATH = f"s3://{BUCKET}/models/churn/xgboost/version=1/"


# 1. Get the absolute path of the directory where run_training.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Join it to the subfolder containing your code
# This ensures it works whether you run from CMD, PowerShell, or PyCharm
#training_code_dir = os.path.join(BASE_DIR, "src", "training")

logger.info(f"Looking for training code in: {BASE_DIR}")


boto_session = boto3.Session(region_name=REGION)
session = sagemaker.Session(boto_session=boto_session)
try:

    logger.info("training model Started")

    estimator = SKLearn(
        entry_point="train.py",  # The script name only
        source_dir=BASE_DIR,  # The folder containing the script
        role=ROLE,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        output_path=MODEL_PATH,
        sagemaker_session=session  # Use the region-locked session
    )

    estimator.fit(
        {
            "train": TRAIN_PATH,
            "validation": VAL_PATH
        }
    )
    logger.info("training model Completed")
except Exception as e:
    logger.error(e)
    logger.error("Running training model Failed")
