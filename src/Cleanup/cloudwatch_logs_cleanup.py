import boto3

logs_client = boto3.client("logs", region_name="ap-south-1")
log_group_name = "/aws/sagemaker/Endpoints/churn-xgboost-prod"

try:
    # Set retention to 1 day. After 24 hours, the logs disappear automatically.
    logs_client.put_retention_policy(
        logGroupName=log_group_name,
        retentionInDays=1
    )
    print(f"Retention policy set to 1 day for {log_group_name}")
except Exception as e:
    print(f"Could not set retention (log group might not exist yet): {e}")