#!/bin/bash

# ===== CONFIG =====
BUCKET_NAME="ml-prod-pipeline"
REGION="ap-south-1"

# ===== CREATE BUCKET =====
aws s3api create-bucket \
  --bucket $BUCKET_NAME \
  --region $REGION \
  --create-bucket-configuration LocationConstraint=$REGION

# ===== BASE FOLDERS =====
FOLDERS=(
  "raw/sensors/"
  "raw/maintenance_logs/"
  "raw/external/"

  "quarantine/"

  "processed/train/"
  "processed/validation/"
  "processed/test/"

  "features/offline/"
  "features/online/"

  "models/artifacts/"
  "models/registry/"

  "endpoints/logs/"

  "predictions/batch/"
  "predictions/realtime/"

  "monitoring/data_drift/"
  "monitoring/model_quality/"

  "logs/glue/"
  "logs/sagemaker/"
  "logs/stepfunctions/"
)

# ===== CREATE FOLDERS =====
for folder in "${FOLDERS[@]}"; do
  aws s3api put-object \
    --bucket $BUCKET_NAME \
    --key "$folder"
done

echo "✅ S3 production folder structure created successfully"
