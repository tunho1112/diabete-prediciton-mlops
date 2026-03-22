import argparse
import json
import os
import random
import shutil

import numpy as np


def main(args):
    # Clean up the avro schema folder if exists,
    # then create a new folder
    if os.path.exists(args["schema_folder"]):
        shutil.rmtree(args["schema_folder"])
    os.mkdir(args["schema_folder"])


    # Initialize schema template
    schema = {
        "doc": "Diabetes schema to help you get started.",
        "fields": [
            {"name": "diabete_id", "type": "string"},
            {"name": "created", "type": "string"},
            {"name": "Pregnancies", "type": "int"},
            {"name": "Glucose", "type": "int"},
            {"name": "BloodPressure", "type": "int"},
            {"name": "SkinThickness", "type": "int"},
            {"name": "Insulin", "type": "int"},
            {"name": "BMI", "type": "float"},
            {"name": "DiabetesPedigreeFunction", "type": "float"},
            {"name": "Age", "type": "int"}
        ],
        "name": "Diabetes",
        "namespace": "diabetes.avro",
        "type": "record",
    }

    # # Write this schema to the Avro output folder
    with open(f'{args["schema_folder"]}/schema_diabetes.avsc', "w+") as f:
        json.dump(schema, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_schemas",
        default=1,
        type=int,
        help="Number of avro schemas to generate.",
    )
    parser.add_argument(
        "-m",
        "--min_features",
        default=2,
        type=int,
        help="Minumum number of features for each device",
    )
    parser.add_argument(
        "-a",
        "--max_features",
        default=10,
        type=int,
        help="Maximum number of features for each device",
    )
    parser.add_argument(
        "-o",
        "--schema_folder",
        default="./avro_schemas",
        help="Folder containing all generated avro schemas",
    )
    args = vars(parser.parse_args())
    main(args)
