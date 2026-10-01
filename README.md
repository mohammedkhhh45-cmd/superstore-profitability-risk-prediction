# 🛒 Superstore Order Profitability & Loss Risk Prediction

An end-to-end Machine Learning project that shifts Superstore analytics from descriptive dashboarding to predictive risk modeling. This repository contains a complete pipeline that evaluates order quotes in real time to predict whether a transaction will incur a financial loss (`Profit < 0`) before approval.

---

## 🎯 Business Problem
In retail operations, aggressive discounting often destroys product margins. In the **Sample - Superstore** dataset:
* **18.72% of all orders (1,871 transactions)** resulted in a net financial loss.
* Heavy discounts (>20–30%) significantly increased the likelihood of negative profit margins across major product categories.

**Objective:** Build an early-warning machine learning pipeline that flags unprofitable orders at quote time, allowing sales managers to optimize discounting guardrails.

---

## 📊 Key Findings & Insights
* **Primary Loss Driver:** **Discount** is the single largest predictor of order loss, contributing nearly 50% of total feature importance weight.
* **High-Risk Categories:** Sub-categories such as *Tables*, *Bookcases*, and *Binders* exhibit the highest frequency of unprofitable sales when paired with discounts exceeding 20%.
* **Model Performance:** The Random Forest Classifier achieved an **ROC-AUC score of ~0.98** and an **82%+ recall** on loss-making orders, effectively catching most unprofitable quotes.

---

## 🛠️ Technical Stack & Workflow
* **Language:** Python 3.x
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib

### ML Pipeline Architecture
1. **Target Engineering:** Formulated binary classification target `Is_Loss` (`1` if `Profit < 0`, else `0`).
2. **Preprocessing:** Used `ColumnTransformer` with `StandardScaler` for numeric values (`Sales`, `Quantity`, `Discount`) and `OneHotEncoder` for categorical variables (`Ship Mode`, `Segment`, `Region`, `Category`, `Sub-Category`).
3. **Model Suite:** Trained and evaluated Logistic Regression, Random Forest Classifier, and Gradient Boosting.
4. **Serialization:** Exported the complete trained pipeline as `profitability_loss_model.pkl` for real-time inference.

---

## 📁 Repository Structure
```text
[Day 6 capstone project (1).xlsx](https://github.com/user-attachments/files/32935808/Day.6.capstone.project.1.xlsx)    # Raw Superstore Dataset
[Uploading icarr.py…]()  # Complete Colab Notebook (EDA, Training, Evaluation)
       # Serialized Machine Learning Pipeline
├── README.md                          # Project Documentation
## 🚀 How to Run the Inference Model

### Load the Model & Predict in Python
```python
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load('profitability_loss_model.pkl')

# Sample Order Quote
sample_quote = pd.DataFrame([{
    'Ship Mode': 'Standard Class',
    'Segment': 'Consumer',
    'Region': 'East',
    'Category': 'Technology',
    'Sub-Category': 'Phones',
    'Sales': 1500.00,
    'Quantity': 2,
    'Discount': 0.40  # 40% Discount
}])

# Evaluate Risk
loss_prob = model.predict_proba(sample_quote)[0][1]
prediction = model.predict(sample_quote)[0]

print(f"Loss Risk Probability: {loss_prob * 100:.2f}%")
print("Status:", "🔴 HIGH RISK (Unprofitable)" if prediction == 1 else "🟢 LOW RISK (Profitable)")
