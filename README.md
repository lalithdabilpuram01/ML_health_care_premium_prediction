# ML_health_care_premium_prediction

This project is a **Machine Learning application** that predicts health insurance premiums based on user demographics, lifestyle choices, and medical history. The app leverages **Linear Regression** and **XGBoost** models to provide accurate premium predictions and is deployed as an interactive **Streamlit** web app.

---

## 🔹 Project Overview

Health insurance pricing depends on multiple factors such as age, income, BMI, smoking status, and medical history. Traditional approaches often use static formulas, but **Machine Learning enables data-driven, personalized predictions** by capturing non-linear relationships and hidden patterns in the data.

This project demonstrates an **end-to-end ML pipeline** — from data preprocessing and model training to hyperparameter tuning and web app deployment.

---

## 🔹 Features

* Takes **user input** (age, income, dependents, BMI, smoking status, region, employment type, insurance plan, medical history, etc.).
* **Data segmentation** by age group to improve model accuracy.
* Uses **Linear Regression** for interpretability and **XGBoost** for higher predictive performance.
* **RandomizedSearchCV** applied for hyperparameter tuning.
* **Interactive UI** built with **Streamlit**.
* **Deployed** on Streamlit Cloud with source code hosted on GitHub.

---

## 🔹 Tech Stack

**Languages & Libraries**

* Python, Pandas, NumPy, Scikit-learn, Statsmodels, XGBoost, Matplotlib, Seaborn, Joblib

**Frameworks & Tools**

* Streamlit (frontend & deployment)
* GitHub (version control & code hosting)

---

## 🔹 Workflow

1. **Data Preprocessing**

   * Cleaned and transformed dataset
   * Encoded categorical variables
   * Normalized continuous variables

2. **Segmentation**

   * Split dataset by age groups for better risk-based predictions

3. **Model Training**

   * Linear Regression as baseline model
   * XGBoost as advanced model with hyperparameter tuning using RandomizedSearchCV

4. **Model Evaluation**

   * Metrics: RMSE, R² score
   * Compared model performance across age groups

5. **Deployment**

   * Built interactive interface with Streamlit
   * Hosted app on Streamlit Cloud


---

## 🔹 How to Run Locally

Clone the repository:

```bash
git clone https://github.com/your-username/healthcare-premium-prediction.git
cd healthcare-premium-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run main.py
```

---

## 🔹 Project Links

* **Live App:** [Streamlit Cloud Link](#)
* **GitHub Repo:** [GitHub Repo Link](#)

---

## 🔹 Future Improvements

* Add more advanced models (LightGBM, Neural Networks)
* Incorporate real-world datasets for richer predictions
* Improve explainability with SHAP or LIME
* Enhance UI with additional visualizations and insights

---

## 🔹 Author

**Lalith Kumar Dabilpuram**

* [LinkedIn](https://www.linkedin.com/in/lalithkumardabilpuram)
* Email: [lalith.dabilpuram01@gmail.com](mailto:lalith.dabilpuram01@gmail.com)



