# This is an example feature definition file

from datetime import timedelta

from feast import KafkaSource
from feast.data_format import JsonFormat
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import \
    PostgreSQLSource

diabete_batch_source = PostgreSQLSource(
    name="diabetes",
    query="SELECT * FROM diabetes",
    timestamp_field="created",
)

# device_stats_stream_source = KafkaSource(
#     name="device_stats_stream_source",
#     kafka_bootstrap_servers="localhost:9092",
#     topic="diabetes",
#     timestamp_field="created",
#     batch_source=device_stats_batch_source,
#     message_format=JsonFormat(
#         schema_json="created timestamp, diabete_id integer, pregnancies integer, glucose integer, bloodpressure integer, skinthickness integer, insulin integer, bmi double, dpf double, age integer"
#     ),
#     watermark_delay_threshold=timedelta(minutes=1),
# )
