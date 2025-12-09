#  No-Code Machine Learning Pipeline Builder

A web-based, no-code ML pipeline builder built using Streamlit.  
The user can upload a dataset, preprocess features, split data, select a model, and view results — all without writing code.

---

##  Features
✔ Upload CSV/Excel dataset  
✔ Automatic display of data info and preview  
✔ Preprocessing (Standardization / Normalization)  
✔ Train–Test Split with slider control  
✔ Model selection:
- Logistic Regression  
- Decision Tree Classifier  

✔ Performance output:
- Accuracy score  
- Classification report  
- Confusion matrix visualization  

---


## Architecture

```text
no_code_ml_pipeline/
│
├── app.py                     # Streamlit UI
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_loader.py         # Dataset loading & summary
    ├── preprocessing.py       # Feature scaling
    ├── split_data.py          # Encoding & train-test split
    ├── models.py              # Model selection
    └── evaluation.py          # Scoring & visualization-
```


# How to run it
pip install -r requirements.txt
streamlit run app.py
