import os
import random
from datetime import datetime
from time import sleep

from dotenv import load_dotenv
from postgresql_client import PostgresSQLClient

load_dotenv()

TABLE_NAME = "diabetes"
NUM_ROWS = 100


def main():
    pc = PostgresSQLClient(
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    # Get all columns from the diabetes table
    try:
        columns = pc.get_columns(table_name=TABLE_NAME)
        print(columns)
    except Exception as e:
        print(f"Failed to get schema for table with error: {e}")

    # Loop over all columns and create random values
    for _ in range(NUM_ROWS):
        diabete_id = random.randint(0, 100)
        pregnancies = random.randint(0, 15)
        glucose = random.randint(70, 200)
        bloodpressure = random.randint(40, 120)
        skinthickness = random.randint(0, 60)
        insulin = random.randint(0, 300)
        bmi = round(random.uniform(18.0, 45.0), 1)
        dpf = round(random.uniform(0.05, 2.5), 3)
        age = random.randint(18, 80)
        query = f"""
            INSERT INTO {TABLE_NAME} (
                diabete_id,
                pregnancies,
                glucose,
                bloodpressure,
                skinthickness,
                insulin,
                bmi,
                diabetespedigreefunction,
                age
            )
            VALUES (
                {diabete_id},
                {pregnancies},
                {glucose},
                {bloodpressure},
                {skinthickness},
                {insulin},
                {bmi},
                {dpf},
                {age}
            )
        """
        pc.execute_query(query)
        sleep(2)


if __name__ == "__main__":
    main()
