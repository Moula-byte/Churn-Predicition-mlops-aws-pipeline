import boto3
from src.logger import setup_logger

logger=setup_logger("create_glue_table")

glue = boto3.client("glue", region_name="ap-south-1")

DATABASE_NAME = "churn_ml_db"
BUCKET = "ml-prod-pipeline"

TABLES = {
    "churn_train": f"s3://{BUCKET}/processed/model_splits/train/",
    "churn_validation": f"s3://{BUCKET}/processed/model_splits/validation/",
    "churn_test": f"s3://{BUCKET}/processed/model_splits/test/",
}

# Schema based on churnData.csv
COLUMNS = [
    {"Name": "tenure", "Type": "double"},
    {"Name": "age", "Type": "double"},
    {"Name": "address", "Type": "double"},
    {"Name": "income", "Type": "double"},
    {"Name": "ed", "Type": "double"},
    {"Name": "employ", "Type": "double"},
    {"Name": "equip", "Type": "double"},
    {"Name": "callcard", "Type": "double"},
    {"Name": "wireless", "Type": "double"},
    {"Name": "longmon", "Type": "double"},
    {"Name": "tollmon", "Type": "double"},
    {"Name": "equipmon", "Type": "double"},
    {"Name": "cardmon", "Type": "double"},
    {"Name": "wiremon", "Type": "double"},
    {"Name": "longten", "Type": "double"},
    {"Name": "tollten", "Type": "double"},
    {"Name": "cardten", "Type": "double"},
    {"Name": "voice", "Type": "double"},
    {"Name": "pager", "Type": "double"},
    {"Name": "internet", "Type": "double"},
    {"Name": "callwait", "Type": "double"},
    {"Name": "confer", "Type": "double"},
    {"Name": "ebill", "Type": "double"},
    {"Name": "loglong", "Type": "double"},
    {"Name": "logtoll", "Type": "double"},
    {"Name": "lninc", "Type": "double"},
    {"Name": "custcat", "Type": "int"},
    {"Name": "churn", "Type": "int"},
]

for table_name, s3_path in TABLES.items():
    try:
        logger.info("creating Glue Tables Started")
        glue.create_table(
            DatabaseName=DATABASE_NAME,
            TableInput={
                "Name": table_name,
                "TableType": "EXTERNAL_TABLE",
                "Parameters": {
                    "classification": "parquet"
                },
                "StorageDescriptor": {
                    "Columns": COLUMNS,
                    "Location": s3_path,
                    "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                    "SerdeInfo": {
                        "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                        "Parameters": {
                            "serialization.format": "1"
                        }
                    }
                }
            }
        )
        logger.info(f"Table '{table_name}' created")
    except glue.exceptions.AlreadyExistsException:
        logger.error(f"Table '{table_name}' already exists")
