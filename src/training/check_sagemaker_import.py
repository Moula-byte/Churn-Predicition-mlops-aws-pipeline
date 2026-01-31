import sagemaker
import pandas as pd
import mlflow

print(f"SageMaker: {sagemaker.__version__}")
print(f"Pandas: {pd.__version__}") # Should be 2.x.x
print(f"MLflow: {mlflow.__version__}")
print(" All systems go!")