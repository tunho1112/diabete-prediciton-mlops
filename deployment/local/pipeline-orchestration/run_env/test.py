import s3fs
import pyarrow.dataset as ds
from sqlalchemy import create_engine



s3 = s3fs.S3FileSystem(
    key="minio_access_key",
    secret="minio_secret_key",
    client_kwargs={"endpoint_url": "http://localhost:9000"},
)

bucket = "diabetes-data"
prefix = "pump"      # ví dụ: "raw/"

dataset = ds.dataset(
    f"{bucket}/{prefix}",
    filesystem=s3,
    format="parquet"
)
df = dataset.to_table().to_pandas()
df = df.drop(columns=["outcome"])


engine = create_engine(
    "postgresql+psycopg2://k6:k6@localhost:5432/k6"
)

table_name = "diabetes"

df.to_sql(
    "diabetes",
    engine,
    if_exists="replace",
    index=False
)
