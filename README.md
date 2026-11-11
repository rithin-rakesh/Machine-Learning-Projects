# Loan Fraud Detection — Machine Learning Project

A machine-learning experiment and Streamlit interface for exploring **loan application fraud prediction**.

## What is in this repository?

| File | Purpose |
| --- | --- |
| `machine_learning3.ipynb` | Notebook containing data exploration and model-development work. |
| `app2.py` | Streamlit application that collects applicant features and displays a fraud prediction and probability. |

The Streamlit application loads a saved model (`XGModel.sav`) and nine saved categorical encoders. Those model/encoder files are **not present in the repository**. The notebook also refers to `account_data.csv`, which is not included.

## Technology

Python, pandas, scikit-learn-compatible preprocessing, a saved XGBoost-style model, Jupyter Notebook, and Streamlit.

## Run locally

**Requirements:** Python, Streamlit, pandas, and the compatible packages used to train the saved model.

```bash
git clone https://github.com/rithin-rakesh/Machine-Learning-Projects.git
cd Machine-Learning-Projects
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install streamlit pandas scikit-learn xgboost
streamlit run app2.py
```

**Important:** The application will not start successfully until the matching `XGModel.sav` and encoder `.sav` files are supplied. To reproduce the notebook, supply the appropriate `account_data.csv` dataset separately. Never load untrusted pickle/model files.

## Next improvements

- Add a reproducible training pipeline and dependency lockfile.
- Save a documented, versioned preprocessing pipeline together with the model.
- Include evaluation metrics, dataset provenance, and a clearly described train/test split.
- Add automated tests and an example input/output.
- Ensure the model is evaluated for fairness, robustness, and false positives before any real-world use.

## Responsible use

This is an educational/prototype project. Fraud predictions should not be treated as verified findings or used as the sole basis for lending decisions.

## License

No license has been declared in this repository.
