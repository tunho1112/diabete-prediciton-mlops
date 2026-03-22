
from postgresql_client import PostgresSQLClient

def main():
    pc = PostgresSQLClient(
        database="k6",
        user="k6",
        password="k6",
    )

    # Create devices table
    create_table_query = """
        CREATE TABLE IF NOT EXISTS diabetes (
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