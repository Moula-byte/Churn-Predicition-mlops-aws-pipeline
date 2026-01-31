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

def load_parquet(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(f) for f in files])

def main():
    args = parse_args()

    train_df = load_parquet(args.train)
    val_df   = load_parquet(args.validation)

    X_train = train_df.drop(columns=["churn"])
    y_train = train_df["churn"]

    X_val = val_df.drop(columns=["churn"])
    y_val = val_df["churn"]

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
