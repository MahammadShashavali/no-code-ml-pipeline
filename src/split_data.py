import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def encode_target(y: pd.Series):
    """
    Encode target if categorical. Returns (y_encoded, label_encoder_or_None)
    """
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded, le

    return y.values, None


def make_train_test_split(X: pd.DataFrame, y, test_size: float = 0.2):
    """
    Perform train_test_split with stratify.
    Returns:
        X_train, X_test, y_train, y_test, error_message_or_None
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        return X_train, X_test, y_train, y_test, None

    except ValueError as e:
        # Return clean error message instead of crashing the app
        return None, None, None, None, str(e)
