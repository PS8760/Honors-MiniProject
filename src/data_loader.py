"""
Data Loader Module for MindPredict AI
Handles loading, auditing, and merging of mental health datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

FILE_DESCRIPTIONS = {
    "1- mental-illnesses-prevalence.csv": "Prevalence rates of 5 major mental illnesses by country and year",
    "2- burden-disease-from-each-mental-illness(1).csv": "DALY rates (disease burden) for 5 mental illnesses by country and year",
    "3- adult-population-covered-in-primary-data-on-the-prevalence-of-major-depression.csv": "Regional coverage of primary data on major depression prevalence",
    "4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv": "Regional coverage of primary data on multiple mental illnesses",
    "5- anxiety-disorders-treatment-gap.csv": "Treatment gap statistics for anxiety disorders by country",
    "6- depressive-symptoms-across-us-population.csv": "Prevalence of depressive symptoms in US population (2014)",
    "7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv": "Count of countries providing primary data per mental disorder",
}


def get_dataset_path() -> str:
    """Return the path to the dataset directory."""
    return DATA_DIR


def list_csv_files() -> List[str]:
    """List all CSV files in the dataset directory."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    files.sort()
    return files


def load_csv(filename: str) -> pd.DataFrame:
    """Load a single CSV file from the dataset directory."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load all CSV files and return a dictionary of DataFrames."""
    datasets = {}
    for fname in list_csv_files():
        try:
            df = load_csv(fname)
            datasets[fname] = df
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return datasets


def audit_single_file(df: pd.DataFrame, filename: str) -> Dict:
    """Perform a comprehensive audit of a single DataFrame."""
    audit = {
        "filename": filename,
        "description": FILE_DESCRIPTIONS.get(filename, "Unknown dataset"),
        "shape": df.shape,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2) if df.size > 0 else 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
        "unique_entities": df["Entity"].nunique() if "Entity" in df.columns else 0,
        "year_range": None,
        "sample_data": df.head(5).to_dict("records"),
    }

    if "Year" in df.columns:
        audit["year_range"] = (int(df["Year"].min()), int(df["Year"].max()))

    return audit


def full_dataset_audit() -> Dict:
    """Audit all CSV files and return comprehensive results."""
    datasets = load_all_datasets()
    audits = {}
    for fname, df in datasets.items():
        audits[fname] = audit_single_file(df, fname)
    return audits


def get_prevalence_data() -> pd.DataFrame:
    """Load the main prevalence dataset (file 1)."""
    return load_csv("1- mental-illnesses-prevalence.csv")


def get_daly_data() -> pd.DataFrame:
    """Load the DALY burden dataset (file 2)."""
    return load_csv("2- burden-disease-from-each-mental-illness(1).csv")


def merge_prevalence_daly() -> pd.DataFrame:
    """
    Merge prevalence and DALY datasets on Entity, Code, Year.
    Only includes country-level records present in both datasets.
    """
    prev = get_prevalence_data()
    daly = get_daly_data()

    prev_cols = {
        "Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized": "schizophrenia_prev",
        "Depressive disorders (share of population) - Sex: Both - Age: Age-standardized": "depressive_prev",
        "Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized": "anxiety_prev",
        "Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized": "bipolar_prev",
        "Eating disorders (share of population) - Sex: Both - Age: Age-standardized": "eating_prev",
    }
    prev_renamed = prev.rename(columns=prev_cols)
    keep_prev = ["Entity", "Code", "Year"] + list(prev_cols.values())
    prev_renamed = prev_renamed[[c for c in keep_prev if c in prev_renamed.columns]]

    daly_cols = {
        "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders": "depressive_daly",
        "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia": "schizophrenia_daly",
        "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder": "bipolar_daly",
        "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders": "eating_daly",
        "DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders": "anxiety_daly",
    }
    daly_renamed = daly.rename(columns=daly_cols)
    keep_daly = ["Entity", "Code", "Year"] + list(daly_cols.values())
    daly_renamed = daly_renamed[[c for c in keep_daly if c in daly_renamed.columns]]

    merged = pd.merge(prev_renamed, daly_renamed, on=["Entity", "Code", "Year"], how="inner")
    return merged


def get_country_list() -> List[str]:
    """Get sorted list of unique countries from prevalence data."""
    df = get_prevalence_data()
    return sorted(df["Entity"].unique().tolist())


def get_year_range() -> Tuple[int, int]:
    """Get the year range from prevalence data."""
    df = get_prevalence_data()
    return (int(df["Year"].min()), int(df["Year"].max()))


def get_treatment_data() -> pd.DataFrame:
    """Load anxiety disorders treatment gap data."""
    return load_csv("5- anxiety-disorders-treatment-gap.csv")


def get_us_symptoms_data() -> pd.DataFrame:
    """Load US depressive symptoms data."""
    return load_csv("6- depressive-symptoms-across-us-population.csv")


def get_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate an automatic data dictionary from a DataFrame."""
    dict_rows = []
    for col in df.columns:
        dict_rows.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].count()),
            "Missing Count": int(df[col].isnull().sum()),
            "Missing %": round(df[col].isnull().sum() / len(df) * 100, 2),
            "Unique Values": int(df[col].nunique()),
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A",
        })
    return pd.DataFrame(dict_rows)
