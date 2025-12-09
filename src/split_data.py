import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def encode_target(y: pd.Series):
    """
    Encode target column if it is categorical (string).

    Returns:
        y_encoded : array
        label_encoder_or_None
    """
    # if type is object or category → encode
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded, le

    # if already numeric → return as it is
    return y.values, None


def make_train_test_split(X: pd.DataFrame, y, test_size: float = 0.2):
    """
    Perform train-test split with stratify.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
