import numpy as np 
import pandas as pd
import os
from helpers import load_cfg


CFG_PATH = "deployment/datalake/utils/config.yaml"


if __name__ == "__main__":
    df = pd.read_csv("data/diabetes.csv")
    cfg = load_cfg(CFG_PATH)
    data_cfg = cfg["data"]
    num_files = data_cfg["num_files"]

    # Create the destination folder to save
    # parquet files if not existing
    if not os.path.exists(data_cfg["folder_path"]):
        os.makedirs(data_cfg["folder_path"], exist_ok=True)
        # Split data frame by num_files, then write each
    # to a parquet file
    df_splits = np.array_split(df, num_files)
    for i in range(num_files):
        df_splits[i].reset_index().to_parquet(
            os.path.join(data_cfg["folder_path"], f"part_{i}.parquet")
        )