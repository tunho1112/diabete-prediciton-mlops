from kfp import dsl
from kfp import compiler
from kfp.client import Client
from kfp.dsl import Output, Input, Dataset, Model


from kfp import dsl

@dsl.component(
    base_image='python:3.9', 
    packages_to_install=['s3fs', 'pyarrow', 'pandas'])
def data_preparation(
    diabete_dataset: Output[Dataset]
) -> Dataset:
    import s3fs
    import pyarrow.dataset as ds
    import pandas 

    s3 = s3fs.S3FileSystem(
        key="minio_access_key",
        secret="minio_secret_key",
        client_kwargs={"endpoint_url": "http://10.24.1.39:9000"},
    )
    bucket = "diabetes-data"
    prefix = "pump" 
    dataset = ds.dataset(
        f"{bucket}/{prefix}",
        filesystem=s3,
        format="parquet"
    )
    df = dataset.to_table().to_pandas()
    df = df.drop(columns=["index"])
    with open(diabete_dataset.path, 'w') as f:
        df.to_csv(f)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==1.3.5', 'scikit-learn==1.0.2'])
def normalize_dataset(
    diabete_dataset: Input[Dataset],
    normalized_diabete_dataset_train: Output[Dataset],
    diabete_label_train: Output[Dataset],
    normalized_diabete_dataset_test: Output[Dataset],
    diabete_label_test: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    

    with open(diabete_dataset.path) as f:
        df = pd.read_csv(f)

    df_train, df_val= train_test_split(df, test_size=0.4, random_state=42)
    _, df_test = train_test_split(df_val, test_size=0.5, random_state=42)
    
    X_train = df_train.drop(columns=["Outcome"])
    y_train = df_train["Outcome"]
    X_test = df_test.drop(columns=["Outcome"])
    y_test = df_test["Outcome"]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open(normalized_diabete_dataset_train.path, 'w') as f:
        X_train.to_csv(f)
    with open(diabete_label_train.path, 'w') as f:
        y_train.to_csv(f)
    with open(normalized_diabete_dataset_test.path, 'w') as f:
        X_test.to_csv(f)
    with open(diabete_label_test.path, 'w') as f:
        y_test.to_csv(f)

@dsl.component(packages_to_install=['pandas==1.3.5', 'scikit-learn==1.0.2', 'xgboost==1.5.0', 'mlflow==2.22.1'])
def train_model(
    normalized_diabete_dataset_train: Input[Dataset],
    diabete_label_train: Input[Dataset],
    normalized_diabete_dataset_test: Input[Dataset],
    diabete_label_test: Input[Dataset],
    model: Output[Model],
    mlflow_uri: str = "http://10.24.1.39:5000", 
    experiment_name: str = "diabetes_model" 
):
    import pandas as pd
    import xgboost as xgb
    import mlflow
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # init mlflow experiment
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name + "_xgb")
    with open(normalized_diabete_dataset_train.path) as f:
        X_train = pd.read_csv(f)
    with open(diabete_label_train.path) as f:
        y_train = pd.read_csv(f)
    with open(normalized_diabete_dataset_test.path) as f:
        X_test = pd.read_csv(f)
    with open(diabete_label_test.path) as f:
        y_test = pd.read_csv(f)

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    # mlflow logging
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        model, 
        "model", 
        signature=signature,
        registered_model_name="xgb_diabetes_model"  # Register to Model Registry
    )
    mlflow.end_run()

@dsl.pipeline
def my_pipeline():
    data_preparation_task = data_preparation()
    normalized_data_task  = normalize_dataset(data_preparation_task.outputs['diabete_dataset'])
    train_model_task = train_model(
        normalized_data_task.outputs['normalized_diabete_dataset_train'],
        normalized_data_task.outputs['diabete_label_train'],
        normalized_data_task.outputs['normalized_diabete_dataset_test'],
        normalized_data_task.outputs['diabete_label_test'],
    )
    

compiler.Compiler().compile(my_pipeline, 'pipeline.yaml')

client = Client(host='http://10.24.1.39:8080')
run = client.create_run_from_pipeline_package(
    'pipeline.yaml',
    arguments={
        'mlflow_uri': 'http://10.24.1.39:5000',
        'experiment_name': 'diabetes_model'
    },
)