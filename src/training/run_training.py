import sagemaker
import boto3
import os
from sagemaker.sklearn.estimator import SKLearn
from xgboost import XGBClassifier

from src.logger import setup_logger

logger = setup_logger("run_training")



REGION = "ap-south-1"
ROLE = "arn:aws:iam::998331943727:role/SageMakerExecutionRole-MLProd"

BUCKET = "ml-prod-pipeline"

TRAIN_PATH = f"s3://{BUCKET}/processed/model_splits/train/"
VAL_PATH   = f"s3://{BUCKET}/processed/model_splits/validation/"
MODEL_PATH = f"s3://{BUCKET}/models/churn/xgboost/version=1/"


current_dir = os.path.dirname(os.path.abspath(__file__))
#training_code_dir = os.path.join(current_dir, "src", "training")

boto_session = boto3.Session(region_name=REGION)
session = sagemaker.Session(boto_session=boto_session)
try:

    logger.info("training model Started")

    estimator = SKLearn(
        entry_point="train.py",  # The script name only
        source_dir=current_dir,  # The folder containing the script
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