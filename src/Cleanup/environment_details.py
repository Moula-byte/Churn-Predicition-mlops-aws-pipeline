import sys
import platform
import subprocess
import json
import boto3

print("=" * 60)
print("AWS Churn MLOps – Environment & Infra Check")
print("=" * 60)

# -------------------------------
# Python & OS
# -------------------------------
print("\n[1] Python & OS")
print(f"Python Version : {sys.version}")
print(f"Executable     : {sys.executable}")
print(f"OS             : {platform.system()} {platform.release()}")

if sys.version_info[:2] != (3, 10):
    print("⚠️  WARNING: Recommended Python version is 3.10")
else:
    print("✅ Python version OK")

# -------------------------------
# Virtual Environment
# -------------------------------
print("\n[2] Virtual Environment")
if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
    print("✅ Running inside virtual environment")
else:
    print("⚠️  WARNING: Not running inside a virtual environment")

# -------------------------------
# Installed Libraries
# -------------------------------
print("\n[3] Core Library Versions")

def get_version(pkg):
    try:
        return __import__(pkg).__version__
    except Exception:
        return "NOT INSTALLED"

libs = ["boto3", "sagemaker", "sklearn", "pandas", "numpy", "joblib"]
for lib in libs:
    print(f"{lib:<10} : {get_version(lib)}")

# -------------------------------
# AWS CLI
# -------------------------------
print("\n[4] AWS CLI")
try:
    result = subprocess.run(
        ["aws", "--version"],
        capture_output=True,
        text=True
    )
    print(result.stdout.strip() or result.stderr.strip())
except FileNotFoundError:
    print("❌ AWS CLI not found")

# -------------------------------
# AWS Identity & Region
# -------------------------------
print("\n[5] AWS Identity & Region")
try:
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    print("✅ AWS Credentials Found")
    print(json.dumps(identity, indent=2))

    session = boto3.session.Session()
    print(f"Region          : {session.region_name}")
except Exception as e:
    print("❌ AWS credentials not configured")
    print(str(e))

# -------------------------------
# S3 Access Check
# -------------------------------
print("\n[6] S3 Access Check")
try:
    s3 = boto3.client("s3")
    buckets = s3.list_buckets()
    print(f"✅ Accessible Buckets: {len(buckets['Buckets'])}")
except Exception as e:
    print("❌ S3 access failed")
    print(str(e))

# -------------------------------
# SageMaker Availability
# -------------------------------
print("\n[7] SageMaker Service Check")
try:
    sm = boto3.client("sagemaker")
    response = sm.list_training_jobs(MaxResults=1)
    print("✅ SageMaker API accessible")
except Exception as e:
    print("❌ SageMaker access failed")
    print(str(e))

print("\n" + "=" * 60)
print("Environment & Infra Check Completed")
print("=" * 60)
