import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os


def test_local_prediction():
    model_path = "src/local model/model.json"

    if not os.path.exists(model_path):
        print("Model file not found! Make sure you extracted model.tar.gz")
        return

    # 1. Load the model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Model loaded successfully!")

    # 2. Load and print metrics
    with open("src/local model/model.json", "r") as f:
        metrics = json.load(f)
        print(f"Model Training Metrics: {metrics}")

    # 3. Create dummy data (Ensure this matches the number of columns you trained with)
    # We used float32 in training, so we use it here too.
    # Note: Replace '28' with the actual number of features your model expects.
    num_features = model.get_booster().num_features()
    dummy_data = np.random.rand(1, num_features).astype('float32')

    # 4. Predict
    prediction = model.predict(dummy_data)
    probability = model.predict_proba(dummy_data)[:, 1]

    print(f"Input features: {num_features}")
    print(f"Prediction (Churn=1, Stay=0): {prediction[0]}")
    print(f"Churn Probability: {probability[0]:.4f}")


if __name__ == "__main__":
    test_local_prediction()