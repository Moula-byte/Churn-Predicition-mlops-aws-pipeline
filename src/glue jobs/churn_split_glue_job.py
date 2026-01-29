import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, count, lit
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id

# -----------------------------
# CONFIG
# -----------------------------
SNAPSHOT_DATE = "2026-01-29"
# Everything is now under the SAME bucket
BUCKET = "s3://ml-prod-pipeline"

INPUT_PATH = f"{BUCKET}/processed/base_table/snapshot_date={SNAPSHOT_DATE}/churn_base_20260129.csv"
OUTPUT_BASE = f"{BUCKET}/processed/model_splits/"  # New sub-folder for results

spark = SparkSession.builder.appName("ChurnTrainValTestSplit").getOrCreate()

# -----------------------------
# READ
# -----------------------------
print(f"Reading from: {INPUT_PATH}")
df = spark.read.option("header", True).csv(INPUT_PATH)
df = df.withColumn("snapshot_date", lit(SNAPSHOT_DATE))
df = df.withColumn("churn", col("churn").cast("int"))


df = df.withColumn("customer_id", monotonically_increasing_id())
# Then your existing window spec will work:

# -----------------------------
# STRATIFIED SPLIT
# -----------------------------
window_spec = Window.partitionBy("churn").orderBy("customer_id")

count_window = Window.partitionBy("churn")

df = df.withColumn("row_num", row_number().over(window_spec))
df = df.withColumn("group_total", count("*").over(count_window))
df = df.withColumn("split_ratio", col("row_num") / col("group_total"))

train_df = df.filter(col("split_ratio") <= 0.7)
val_df = df.filter((col("split_ratio") > 0.7) & (col("split_ratio") <= 0.85))
test_df = df.filter(col("split_ratio") > 0.85)

# -----------------------------
# WRITE (To same bucket, different prefix)
# -----------------------------
datasets = {"train": train_df, "validation": val_df, "test": test_df}

for name, data in datasets.items():
    # Helper columns dropped during write
    final_data = data.drop("row_num", "group_total", "split_ratio")

    output_path = f"{OUTPUT_BASE}{name}/snapshot_date={SNAPSHOT_DATE}/"
    final_data.write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved {name} set to {output_path}")