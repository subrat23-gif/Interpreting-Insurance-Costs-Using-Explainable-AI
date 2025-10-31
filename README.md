# 🧾 Interpreting Insurance Costs Using Explainable AI

This project builds a regression model to estimate **medical insurance costs** based on demographic and lifestyle features, and then applies **Explainable AI (XAI)** methods to interpret model behavior and feature influence.

---

## 📌 Objective

- Predict insurance charges using machine learning models
- Improve transparency using explainability tools
- Identify key factors affecting insurance premiums
- Provide both **global** and **local** feature explanations

---

## 📊 Dataset

The dataset includes the following features:

| Feature | Description |
|--------|------------|
age | Age of the individual |
sex | Male / Female |
bmi | Body Mass Index (obesity indicator) |
children | Number of dependents |
smoker | Smoking status |
region | Residential region |
charges | Medical insurance cost (target variable) |

---

## 🧠 Methods Used

### ✅ Machine Learning
- Data preprocessing & feature encoding
- Train-test split
- Regression model training (Linear Regression / Tree-based, etc.)
- Model evaluation (MSE, MAE, RMSE, R² Score)

### ✅ Explainability (XAI)
- **SHAP values** – global & local interpretation
- **Feature importance**
- **Partial dependence / interaction understanding**
- **Residual analysis**

---

## 📈 Evaluation Metrics

| Metric | Explanation |
|-------|------------|
MAE | Mean Absolute Error |
MSE | Mean Squared Error |
R² Score | Coefficient of Determination |

These metrics help verify both accuracy and model reliability.

---

## 🚀 How to Run

### **1. Install Dependencies**
```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn
jupyter notebook "INTERPRETING INSURANCE COSTS USING EXPLAINABLE AI.ipynb"

The notebook will:

Load data

Train model

Generate predictions

Visualize SHAP values and feature effects
📁 project-folder
 ├── INTERPRETING INSURANCE COSTS USING EXPLAINABLE AI.ipynb
 ├── README.md
 └── data (optional)
      └── insurance.csv
Outputs

Model training results

Performance metrics

SHAP plots (feature contributions)

Feature importance chart

Residual error distribution

🛠️ Tech Stack

Python

Pandas / NumPy

Scikit-Learn

SHAP

Matplotlib / Seaborn

Jupyter Notebook / Google Colab

📌 Future Improvements

Add XGBoost / CatBoost models

Hyperparameter tuning (Optuna / GridSearch)

Streamlit or FastAPI demo app

Save trained model (.pkl)

Interactive SHAP dashboard

📜 License

This project is for educational and research purposes.

🤝 Contributions

Feel free to contribute by opening issues or PRs.

⭐ Acknowledgements

Dataset: Insurance cost dataset commonly used for ML practice
SHAP: Lundberg & Lee (NIPS, 2017)
