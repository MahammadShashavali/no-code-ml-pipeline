from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_model(model_name: str):
    """
    Return a scikit-learn model instance based on user's choice.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)

    if model_name == "Decision Tree Classifier":
        return DecisionTreeClassifier(random_state=42)

    # fallback (should not happen)
    return LogisticRegression(max_iter=1000)
