from datetime import timedelta

from data_sources import diabete_batch_source
from entities import diabete
from feast import FeatureView, Field
from feast.types import Float32, Int32
# from feast.stream_feature_view import stream_feature_view
# from pyspark.sql import DataFrame

diabete_view = FeatureView(
    name="diabete",
    description="diabete features",
    entities=[diabete],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="pregnancies", dtype=Int32),
        Field(name="glucose", dtype=Int32),
        Field(name="pregnancies", dtype=Int32),
        Field(name="skinthickness", dtype=Int32),
        Field(name="insulin", dtype=Int32),
        Field(name="bmi", dtype=Float32),
        Field(name="diabetespedigreefunction", dtype=Float32),
        Field(name="age", dtype=Int32)
    ],
    online=True,
    source=diabete_batch_source,
)


# @stream_feature_view(
#     entities=[device],
#     ttl=timedelta(days=36500),
#     mode="spark",
#     schema=[
#         Field(name="feature_5", dtype=Float32),
#         Field(name="feature_3", dtype=Float32),
#         Field(name="feature_1", dtype=Float32),
#         Field(name="feature_8", dtype=Float32),
#         Field(name="feature_6", dtype=Float32),
#         Field(name="feature_0", dtype=Float32),
#         Field(name="feature_4", dtype=Float32),
#     ],
#     timestamp_field="created",
#     online=True,
#     source=device_stats_stream_source,
# )
# def device_stats_stream(df: DataFrame):
#     from pyspark.sql.functions import col

#     return (
#         df.withColumn("new_feature_5", col("feature_5") + 1.0)
#         .withColumn("new_feature_3", col("feature_3") + 1.0)
#         .withColumn("new_feature_1", col("feature_1") + 1.0)
#         .withColumn("new_feature_8", col("feature_8") + 1.0)
#         .withColumn("new_feature_6", col("feature_6") + 1.0)
#         .withColumn("new_feature_0", col("feature_0") + 1.0)
#         .withColumn("new_feature_4", col("feature_4") + 1.0)
#         .drop(
#             "feature_5",
#             "feature_3",
#             "feature_1",
#             "feature_8",
#             "feature_6",
#             "feature_0",
#             "feature_4",
#         )
#         .withColumnRenamed("new_feature_5", "feature_5")
#         .withColumnRenamed("new_feature_3", "feature_3")
#         .withColumnRenamed("new_feature_1", "feature_1")
#         .withColumnRenamed("new_feature_8", "feature_8")
#         .withColumnRenamed("new_feature_6", "feature_6")
#         .withColumnRenamed("new_feature_0", "feature_0")
#         .withColumnRenamed("new_feature_4", "feature_4")
#     )
