from kfp import dsl
from kfp import compiler
from kfp.client import Client
from kfp.dsl import Output, Input, Dataset


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
    normalized_diabete_dataset: Output[Dataset],
    diabete_label: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split


    with open(diabete_dataset.path) as f:
        df = pd.read_csv(f)

    df_train, _= train_test_split(df, test_size=0.4, random_state=42)

    X_train = df_train.drop(columns=["Outcome"])
    y_train = df_train["Outcome"]

    # scaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
 
    with open(normalized_diabete_dataset.path, 'w') as f:
        X_train.to_csv(f)
    with open(diabete_label.path, 'w') as f:
        y_train.to_csv(f)


@dsl.pipeline
def my_pipeline():
    data_source = data_preparation()
    data = normalize_dataset(data)

compiler.Compiler().compile(my_pipeline, 'pipeline.yaml')

client = Client(host='http://10.24.1.39:8080')
run = client.create_run_from_pipeline_package(
    'pipeline.yaml',
    # arguments={
    #     'recipient': 'World',
    # },
)