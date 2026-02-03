import json
import boto3
import sagemaker
from sagemaker.predictor import Predictor

# 1. Use the ACTUAL name from your deployment
endpoint_name = "churn-xgboost-prod"
REGION = "ap-south-1"


boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client("sagemaker")


# 2. Initialize the Predictor
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session
)

# 3. Setup Serializers (Better than manual json.dumps)
predictor.serializer = sagemaker.serializers.JSONSerializer()
predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

# 4. Correct Sample Input (Must be 28 features)
# Order: tenure, age, address, income, ed, employ, equip, callcard, wireless,
# longmon, tollmon, equipmon, cardmon, wiremon, longten, tollten, cardten,
# voice, pager, internet, callwait, confer, ebill, loglong, logtoll,
# lninc, custcat, customer_id
sample_input = {
    "features": [
        12, 45, 1, 55.0, 2, 10, 1, 1, 0,  # tenure through wireless
        15.5, 20.0, 0.0, 15.0, 0.0,       # longmon through wiremon
        250.0, 300.0, 150.0,              # longten through cardten
        0, 0, 1, 1, 1, 0,                 # voice through confer
        4.5, 3.2, 3.9, 2.0, 1             # ebill through customer_id
    ]
}

try:
    # No need for json.dumps() because we added the JSONSerializer above
    response = predictor.predict(sample_input)
    print("Prediction Result:", response)
except Exception as e:
    print(f"Prediction failed: {e}")