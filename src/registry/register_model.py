import os
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from src.logger import setup_logger

logger = setup_logger("register_model")

REGION = "ap-south-1"
ROLE = "arn:aws:iam::998331943727:role/SageMakerExecutionRole-MLProd"
BUCKET = "ml-prod-pipeline"
MODEL_PACKAGE_GROUP = "churn-xgboost-models"

boto_session = boto3.Session(region_name=REGION)
session = sagemaker.Session(boto_session=boto_session)

MODEL_ARTIFACT = (
    f"s3://{BUCKET}/models/churn/xgboost/version=1/sagemaker-scikit-learn-2026-02-02-10-56-37-905/output/model.tar.gz"
)
s3_metrics_path = (
    f"s3://{BUCKET}/models/churn/xgboost/version=1/sagemaker-scikit-learn-2026-02-02-10-56-37-905/output/metrics/metrics.json"
)

# 4. Define Model with Inference Script
# This tells SageMaker: "Use this Python script to handle the model.json"
model = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=REGION,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge"
    ),
    model_data=MODEL_ARTIFACT,
    entry_point="inference.py",       # The name of your script
    source_dir="D:/prctice/ML/AWS Churn Predicition MLops project/src/inference/",        # The folder where inference.py lives
    role=ROLE,
    sagemaker_session=session
)

metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=s3_metrics_path,
        content_type="application/json"
    )
)

# 6. Register
try:
    model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        model_metrics=metrics,
        approval_status="PendingManualApproval"
    )
    logger.info(f"Model registered successfully: Version 2 (with inference script)")
except Exception as e:
    logger.error(f"Failed to register model: {str(e)}")