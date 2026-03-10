import os

from dotenv import load_dotenv
from postgresql_client import PostgresSQLClient

load_dotenv()


def main():
    pc = PostgresSQLClient(
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    # Create devices table
    create_table_query = """
        CREATE TABLE IF NOT EXISTS diabetes (
        created TIMESTAMPTZ DEFAULT NOW(),
        diabete_id INTEGER,
        pregnancies INTEGER,
        glucose INTEGER,
        bloodpressure INTEGER,
        skinthickness INTEGER,
        insulin INTEGER,
        bmi DOUBLE PRECISION,
        diabetespedigreefunction DOUBLE PRECISION,
        age INTEGER
    );  
    """

    try:
        pc.execute_query(create_table_query)
    except Exception as e:
        print(f"Failed to create table with error: {e}")


if __name__ == "__main__":
    main()
