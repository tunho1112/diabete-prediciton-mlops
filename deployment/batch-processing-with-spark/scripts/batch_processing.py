from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
import argparse


def main(args):
    # The entrypoint to access all functions of Spark
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Python Spark read parquet example")
        .getOrCreate()
    )

    # Read from a parquet file (transformation)
    # https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html
    df = spark.read.parquet(args.input)

    # Add a new column 'GroupAge' based on the 'Age' column
    df = df.withColumn(
        "GroupAge", 
        when(df["Age"] < 25, "Young").otherwise("Old")
    )
    # Set the number of partitions with repartitions
    num_partitions = 4  # Replace with the desired number of partitions
    df = df.repartition(
        num_partitions
    )  # Default using hash-based partition, which can potentially be lead to skew problem!
  # Write to a single file result.parquet
    df.coalesce(1).write.parquet("deployment/datawarehouse/data/diabetes")

    # Show data (action)
    df.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="deployment/datalake/data/part_0.parquet",
        help="Data file in parquet format.",
    )
    args = parser.parse_args()
    main(args)
