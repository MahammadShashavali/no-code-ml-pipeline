import pandas as pd


def load_dataset(uploaded_file):
    try:
        filename = uploaded_file.name.lower()

        # Read CSV
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # Read .xlsx using openpyxl
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Read .xls using xlrd
        elif filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")

        else:
            return None, "❌ Unsupported file format. Please upload CSV or Excel."

        if df.empty:
            return None, "❌ The uploaded file is empty."

        return df, None

    except Exception as e:
        return None, f"❌ Failed to read file.\n\nDetails: {e}"
def get_basic_info(df: pd.DataFrame):
    """
    Returns rows, columns, column_names from DataFrame.
    """
    rows, cols = df.shape
    column_names = df.columns
    return rows, cols, column_names