import kfp
from kfp import dsl #Domain specific language- Helps us to define pipeline components(steps)

def data_processing_op():
    return dsl.ContainerOp(
        name = "Data Processing",
        image = "ay2728850/corectal-prediction:latest",
        command = ["python", "src/data_processing.py"]
    )

def model_training_op():
    return dsl.ContainerOp(
        name = "Model Training",
        image = "ay2728850/corectal-prediction:latest",
        command = ["python", "src/model_training.py"]
    )


## Pipeline starts here...

@dsl.pipeline(
    name="MLOPS pipeline",
    description="kubeflow testing pipeline.",
)
def mlops_pipeline():
    data_processing = data_processing_op()
    model_training = model_training_op().after(data_processing)


### RUN 
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        mlops_pipeline, "mlops_pipeline.yaml"
    )
