import sagemaker
import inspect

print("Imported sagemaker from:", inspect.getfile(sagemaker))
print("SageMaker module:", sagemaker)

from sagemaker.estimator import Estimator
print("Estimator import OK")
