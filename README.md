# Diabetes Diagnosis Prediction with Dashboard

A comprehensive machine learning project to predict diabetes using health data. Includes model comparison, interpretability via SHAP, and a Streamlit dashboard for interactive analysis.

## Highlights

- Uses Logistic Regression, Random Forest, XGBoost, and SVM
- SHAP-based explainability
- Interactive Streamlit dashboard
- Ready for deployment or demonstration

## Folder Structure

- `data/` - Dataset (add `diabetes.csv` from Kaggle)
- `notebooks/` - EDA and experimentation
- `src/` - Training, evaluation, and SHAP explanation scripts
- `models/` - Saved model files
- `reports/` - Visual outputs and logs
- `dashboard/` - Streamlit app

## How to Run

1. Place `diabetes.csv` in the `data/` folder.
2. Train models: `python src/train_models.py`
3. Launch dashboard: `streamlit run dashboard/app.py`

