import os
import random
from datetime import datetime
from time import sleep

from dotenv import load_dotenv
from postgresql_client import PostgresSQLClient
import pandas as pd

load_dotenv()

TABLE_NAME = "diabetes"
NUM_ROWS = 100
POSTGRES_DB = "k6"
POSTGRES_USER = "k6"
POSTGRES_PASSWORD = "k6"
PATH_DATA = "data/train.csv"

def main():
    pc = PostgresSQLClient(
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )

    # Get all columns from the diabetes table
    try:
        columns = pc.get_columns(table_name=TABLE_NAME)
        print(columns)
    except Exception as e:
        print(f"Failed to get schema for table with error: {e}")
    df = pd.read_csv(PATH_DATA)
    for index, row in df.iterrows():
        diabete_id = random.randint(0, 100)
        pregnancies = row['Pregnancies']
        glucose = row['Glucose']
        bloodpressure = row['BloodPressure']
        skinthickness = row['Skinthickness']
        insulin = row['Insulin']
        bmi = row['BMI']
        dpf = row['DiabetesPedigreeFunction']
        age = row['Age']

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


if __name__ == "__main__":
    main()
