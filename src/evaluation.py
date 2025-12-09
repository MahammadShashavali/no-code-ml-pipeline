import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test):
    """
    Predict and compute metrics.
    Returns: (y_pred, metrics_dict)
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
    }
    return y_pred, metrics


def plot_confusion_matrix(model, X_test, y_test):
    """
    Returns a matplotlib Figure object with confusion matrix plotted.
    """
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    return fig
