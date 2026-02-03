import os
import json
import xgboost as xgb
import pandas as pd

# This list must match the EXACT order from your previous error log
FEATURE_NAMES = [
    'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
    'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
    'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
    'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
    'lninc', 'custcat', 'customer_id'
]


def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "model.json"))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Extract the list from the "features" key
        data_list = data.get("features", data)

        # KEY FIX: Convert to DataFrame with Column Names
        df = pd.DataFrame([data_list], columns=FEATURE_NAMES)
        return xgb.DMatrix(df)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())