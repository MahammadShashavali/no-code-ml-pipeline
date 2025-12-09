import streamlit as st
import graphviz

from src.data_folder import load_dataset, get_basic_info
from src.preprocessing import get_numeric_columns, apply_preprocessing
from src.split_data import encode_target, make_train_test_split
from src.models import get_model
from src.evaluation import evaluate_model, plot_confusion_matrix


# --------- PAGE SETUP ---------
st.set_page_config(
    page_title="No-Code ML Pipeline Builder",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 No-Code ML Pipeline Builder")
st.caption("Upload → Preprocess → Split → Model → Results (no-code pipeline).")


# --------- SESSION STATE ---------
if "df" not in st.session_state:
    st.session_state.df = None

if "target_col" not in st.session_state:
    st.session_state.target_col = None

if "preprocessing" not in st.session_state:
    st.session_state.preprocessing = "None"

if "X_processed" not in st.session_state:
    st.session_state.X_processed = None
    st.session_state.numeric_cols = None

if "test_size" not in st.session_state:
    st.session_state.test_size = 0.2

if "X_train" not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None

if "model" not in st.session_state:
    st.session_state.model = None


# --------- STEP 1: UPLOAD DATASET ---------
st.header("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: .csv, .xlsx, .xls"
)

if uploaded_file is not None:
    df, error_msg = load_dataset(uploaded_file)

    if error_msg:
        st.error(error_msg)
    else:
        # RESET state when a new file is uploaded
        st.session_state.df = df
        st.session_state.target_col = None
        st.session_state.X_processed = None
        st.session_state.numeric_cols = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None

        rows, cols, column_names = get_basic_info(df)

        st.success("✅ File uploaded successfully!")
        st.write(f"**Rows:** {rows} | **Columns:** {cols}")
        st.write("**Column Names:**", list(column_names))

        with st.expander("👀 Preview dataset"):
            st.dataframe(df.head())

else:
    st.info("Please upload a CSV or Excel file to begin.")

st.markdown("---")

# If no data yet, stop here
if st.session_state.df is None:
    st.stop()

df = st.session_state.df
# --------- PIPELINE DIAGRAM ---------
st.subheader("🔗 Pipeline Overview")

dot = graphviz.Digraph()

# nodes in visual order
dot.node("A", "📂 Data Upload")
dot.node("B", "🧹 Preprocessing")
dot.node("C", "✂️ Train–Test Split")
dot.node("D", "🧠 Model Training")
dot.node("E", "📊 Results")

# connections (flow)
dot.edges(["AB", "BC", "CD", "DE"])

st.graphviz_chart(dot, use_container_width=True)
st.markdown("---")



# --------- STEP 2: DATA PREPROCESSING ---------
st.header("2️⃣ Data Preprocessing")

# 2.1 choose target column
# Safely choose default index
if st.session_state.target_col in df.columns:
    default_index = df.columns.get_loc(st.session_state.target_col)
else:
    default_index = len(df.columns) - 1  # last column

target_col = st.selectbox(
    "Select the target column (what you want to predict)",
    options=df.columns,
    index=default_index,
)
st.session_state.target_col = target_col


#st.session_state.target_col = target_col

# Features & target
X_full = df.drop(columns=[target_col])
y = df[target_col]

# 2.2 find numeric feature columns
numeric_cols = get_numeric_columns(X_full)
st.session_state.numeric_cols = numeric_cols

if not numeric_cols:
    st.error("No numeric columns found for preprocessing. Please upload a dataset with numeric features.")
    st.stop()

st.write("**Numeric feature columns used for the model:**", numeric_cols)

# 👉 USE ONLY NUMERIC COLUMNS FOR X
X_numeric = X_full[numeric_cols]

# 2.3 user chooses preprocessing method
preprocessing_choice = st.radio(
    "Choose a preprocessing step for numeric features",
    ["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)"],
    index=["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)"].index(
        st.session_state.preprocessing
    ),
    help="Only numeric columns are scaled; non-numeric columns are not used in the model."
)
st.session_state.preprocessing = preprocessing_choice

# 2.4 apply preprocessing on numeric features only
X_processed, scaler = apply_preprocessing(X_numeric, numeric_cols, preprocessing_choice)
st.session_state.X_processed = X_processed

st.success(f"✅ Preprocessing applied: {preprocessing_choice}")
with st.expander("🔍 View processed feature sample"):
    st.dataframe(X_processed.head())

st.markdown("---")


# --------- STEP 3: TRAIN–TEST SPLIT ---------
st.header("3️⃣ Train–Test Split")

split_ratio = st.slider(
    "Select test size (percentage of data used for testing)",
    min_value=10,
    max_value=40,
    value=int(st.session_state.test_size * 100),
    step=5
)
test_size = split_ratio / 100.0
st.session_state.test_size = test_size

st.write(f"➡️ Train : Test = {int((1 - test_size) * 100)} : {int(test_size * 100)}")

# encode target if needed
y_encoded, label_encoder = encode_target(y)

# split using only numeric processed features
# split using only numeric processed features
X_train, X_test, y_train, y_test, split_error = make_train_test_split(
    X_processed, y_encoded, test_size=test_size
)

if split_error is not None:
    st.error("⚠️ Could not create a train–test split with the current settings.")
    st.write(
        "This usually happens when some classes in the target column have very few rows.\n\n"
        "**Details:**"
    )
    st.code(split_error)

    st.info(
        "💡 Try one of these:\n"
        "- Choose a different target column\n"
        "- Reduce the test size (e.g., 10–20%)\n"
        "- Remove very rare classes from the dataset before uploading"
    )
    st.stop()

# save into session_state for next steps
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test

st.success(f"✅ Dataset split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

with st.expander("📊 View train & test shapes"):
    st.write("X_train:", X_train.shape)
    st.write("X_test:", X_test.shape)
    st.write("y_train:", y_train.shape)
    st.write("y_test:", y_test.shape)

st.markdown("---")


# --------- STEP 4: MODEL SELECTION ---------
st.header("4️⃣ Model Selection")

model_choice = st.radio(
    "Choose a model to train",
    ["Logistic Regression", "Decision Tree Classifier"],
)
st.session_state.model_name = model_choice

st.markdown("**Model Description:**")
if model_choice == "Logistic Regression":
    st.info("Logistic Regression is a linear model used for classification problems.")
else:
    st.info("Decision Tree Classifier is a tree-based model that splits data based on feature values.")

st.markdown("---")


# --------- STEP 5: TRAIN MODEL & VIEW RESULTS ---------
st.header("5️⃣ Train Model & View Results")

train_button = st.button("🚀 Run Pipeline & Train Model")

if train_button:
    if any(v is None for v in [X_train, X_test, y_train, y_test]):
        st.error("Train–test split not completed. Please check previous steps.")
        st.stop()

    with st.spinner("Training model..."):
        model = get_model(model_choice)
        model.fit(X_train, y_train)
        st.session_state.model = model

        y_pred, metrics = evaluate_model(model, X_test, y_test)

        st.success("✅ Model trained successfully!")

        st.subheader("📈 Performance Metrics")
        st.metric(label="Accuracy", value=f"{metrics['accuracy']*100:.2f}%")

        st.write("**Detailed Classification Report:**")
        st.text(metrics["classification_report"])

        st.subheader("🔢 Confusion Matrix")
        fig_cm = plot_confusion_matrix(model, X_test, y_test)
        st.pyplot(fig_cm)
        results_df = X_test.copy()
        results_df["Actual"] = y_test
        results_df["Predicted"] = y_pred

# Download as CSV button
        st.subheader("📥 Download Results")
        st.download_button(
            label="Download Predictions as CSV",
            data=results_df.to_csv(index=False),
            file_name="model_predictions.csv",
            mime="text/csv")

        st.subheader("✅ Pipeline Execution Status")
        st.success(
            f"Pipeline completed: Data → Preprocessing ({preprocessing_choice}) → "
            f"Split ({int((1-test_size)*100)}/{int(test_size*100)}) → {model_choice} → Results"
        )
else:
    st.info("Click **'Run Pipeline & Train Model'** after selecting a model.")
