"""
Downloads the official UCI Bank Marketing dataset (Dataset ID 222, Moro et al., 2014)
and formats `data/bank_marketing.csv` with a deterministic `row_id` column and
clean column names (replacing dots with underscores for Spark SQL compatibility).

Usage on an internet-connected machine:
    python src/download_bank_data.py
"""
import io
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bank_marketing.csv")

def download_and_prepare():
    print(f"Downloading UCI Bank Marketing dataset from {UCI_URL} ...")
    req = urllib.request.urlopen(UCI_URL, timeout=60)
    outer_zip = zipfile.ZipFile(io.BytesIO(req.read()))
    inner_bytes = outer_zip.read("bank-additional.zip")
    inner_zip = zipfile.ZipFile(io.BytesIO(inner_bytes))
    with inner_zip.open("bank-additional/bank-additional-full.csv") as csv_file:
        df = pd.read_csv(csv_file, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df["y"] = (df["y"].astype(str).str.lower().isin(["yes", "1", "true"])).astype(int)
    df.insert(0, "row_id", np.arange(1, len(df) + 1))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns) - 2} features to {OUT_PATH}")

if __name__ == "__main__":
    download_and_prepare()
