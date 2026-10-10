"""
Data Loader Module for Student Performance Dataset
Dataset Source: UCI Machine Learning Repository (Cortez and Silva, 2008)
"""

import os
import urllib.request
import zipfile
import pandas as pd

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download_and_extract(data_dir: str = "data") -> None:
    """Downloads and extracts the dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    mat_path = os.path.join(data_dir, "student-mat.csv")
    por_path = os.path.join(data_dir, "student-por.csv")

    if not (os.path.exists(mat_path) and os.path.exists(por_path)):
        print(f"Dataset not found locally. Downloading from {UCI_DATA_URL}...")
        zip_path = os.path.join(data_dir, "student.zip")
        urllib.request.urlretrieve(UCI_DATA_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Handle nested zip if present
        nested_zip = os.path.join(data_dir, "student.zip")
        if os.path.exists(nested_zip):
            try:
                with zipfile.ZipFile(nested_zip, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
            except Exception:
                pass
            if os.path.exists(nested_zip):
                os.remove(nested_zip)

        print("Dataset downloaded and extracted successfully.")


def load_dataset(subject: str = "mat", data_dir: str = "data") -> pd.DataFrame:
    """
    Loads student performance data for a given subject.
    
    Parameters:
        subject (str): 'mat' for Mathematics, 'por' for Portuguese, or 'both'.
        data_dir (str): Directory containing the CSV files.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    download_and_extract(data_dir)

    if subject == "mat":
        df = pd.read_csv(os.path.join(data_dir, "student-mat.csv"), sep=";")
        df["subject"] = "Math"
    elif subject == "por":
        df = pd.read_csv(os.path.join(data_dir, "student-por.csv"), sep=";")
        df["subject"] = "Portuguese"
    elif subject == "both":
        df_mat = pd.read_csv(os.path.join(data_dir, "student-mat.csv"), sep=";")
        df_mat["subject"] = "Math"
        df_por = pd.read_csv(os.path.join(data_dir, "student-por.csv"), sep=";")
        df_por["subject"] = "Portuguese"
        df = pd.concat([df_mat, df_por], ignore_index=True)
    else:
        raise ValueError("Subject must be 'mat', 'por', or 'both'")

    return df


if __name__ == "__main__":
    df = load_dataset("mat")
    print(f"Loaded Math Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))
