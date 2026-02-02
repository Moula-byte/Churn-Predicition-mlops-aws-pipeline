import argparse
import pandas as pd
import xgboost as xgb
import os
import json
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_args()

'''
def load_parquet(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(f) for f in files])
'''



def load_parquet(path):
    print(f"Checking directory: {path}")
    try:
        all_files = os.listdir(path)
        print(f"Files found in {path}: {all_files}")
    except Exception as e:
        print(f"Error accessing {path}: {e}")
        return pd.DataFrame()  # Return empty to avoid crash here

    files = [os.path.join(path, f) for f in all_files if f.endswith(".parquet")]

    if not files:
        print(f"WARNING: No .parquet files found in {path}!")
        raise ValueError(f"No parquet files found in {path}. Contents: {all_files}")

    return pd.concat([pd.read_parquet(f) for f in files])



def main():
    args = parse_args()

    train_df = load_parquet(args.train)
    val_df = load_parquet(args.validation)

    # 1. DROP IDENTIFIED NON-NUMERIC COLUMNS
    # Explicitly drop the date column that caused the crash
    cols_to_drop = ['snapshot_date']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])

    # 2. SELECT ONLY NUMERIC-FRIENDLY DATA
    # We convert objects to numeric where possible, and drop the rest
    for df in [train_df, val_df]:
        for col in df.columns:
            if col != 'churn':
                # Try to convert to numeric, non-convertible stuff becomes NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. FINAL CLEANUP
    # Drop any columns that are now entirely NaN (columns that were pure text)
    # Find which columns are numeric enough to keep in training
    train_df = train_df.dropna(axis=1, how='all')

    # Ensure validation matches training columns EXACTLY
    # This prevents errors if validation had extra strings or was missing a column
    val_df = val_df.reindex(columns=train_df.columns, fill_value=0)

    # Now cast safely
    train_df = train_df.astype('float32')
    val_df = val_df.astype('float32')

    # 4. SPLIT
    X_train = train_df.drop(columns=["churn"])
    y_train = train_df["churn"].astype(int)  # XGBoost likes integer labels

    X_val = val_df.drop(columns=["churn"])
    y_val = val_df["churn"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    # Save model
    model_path = os.path.join(args.model_dir, "model.json")
    model.save_model(model_path)

    # Save metrics
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump({"validation_auc": auc}, f)

    print(f"Validation AUC: {auc}")

if __name__ == "__main__":
    main()
