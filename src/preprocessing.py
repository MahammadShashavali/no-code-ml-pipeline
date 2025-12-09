import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_numeric_columns(X: pd.DataFrame):
    """
    Return list of numeric column names.
    """
    return X.select_dtypes(include=[np.number]).columns.tolist()


def apply_preprocessing(X: pd.DataFrame, numeric_cols, method: str):
    """
    Apply chosen preprocessing method to numeric columns.
    Returns (X_processed, scaler_or_None)
    """
    X_processed = X.copy()

    if method == "Standardization (StandardScaler)":
        scaler = StandardScaler()
        X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
        return X_processed, scaler

    if method == "Normalization (MinMaxScaler)":
        scaler = MinMaxScaler()
        X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
        return X_processed, scaler

    # No preprocessing
    return X_processed, None
