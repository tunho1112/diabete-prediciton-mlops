from airflow.decorators import task
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow import DAG


with DAG(dag_id="transform_data", start_date=datetime(2025, 12, 16), schedule=None) as dag:
    @task.virtualenv(
        task_id="minio_to_postgresql",
        requirements=["s3fs", "pyarrow==22.0.0", "sqlalchemy==2.0.45", "pandas==2.3.3", "psycopg2-binary"],
        system_site_packages=False,
    )
    def transform_data():
        import s3fs
        import pyarrow.dataset as ds
        from sqlalchemy import create_engine
        s3 = s3fs.S3FileSystem(
            key="minio_access_key",
            secret="minio_secret_key",
            client_kwargs={"endpoint_url": "http://minio:9000"},
        )

        bucket = "diabetes-data"
        prefix = "pump"      # ví dụ: "raw/"

        dataset = ds.dataset(
            f"{bucket}/{prefix}",
            filesystem=s3,
            format="parquet"
        )
        df = dataset.to_table().to_pandas()
        df = df.drop(columns=["Outcome"])


        engine = create_engine(
            "postgresql+psycopg2://k6:k6@offline-fs:5432/k6"
        )


        df.to_sql(
            "diabetes",
            engine,
            if_exists="replace",
            index=False
        )

    transform_data()

