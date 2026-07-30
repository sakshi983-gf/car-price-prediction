# 🚗 Used Car Price Predictor

A machine learning web app that predicts the fair resale price of used cars using an XGBoost regression model trained on real-world CarDekho listing data.

**🔗 Live Demo:** [your-app-name.streamlit.app](#) *(update this link after deployment)*

---

## 📌 Problem Statement

The used car market in India suffers from price inconsistency, as valuations are often based on subjective dealer judgment rather than data-driven analysis. This project builds a machine learning model that estimates a car's fair market value based on measurable attributes like age, mileage, engine specs, and brand — replacing guesswork with an objective, reproducible valuation.

## 🎯 Solution Overview

- Cleaned and processed 15,000+ real used car listings from CarDekho
- Engineered features like `km_driven_per_year` and `power_to_engine_ratio`
- Compared 6 regression models (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost)
- Tuned the best model (XGBoost) using RandomizedSearchCV
- Achieved **R² ≈ 0.95** on held-out test data
- Deployed as an interactive Streamlit web application

## 🏗️ System Architecture

```
CarDekho Dataset (CSV)
        ↓
Data Preprocessing & Feature Engineering
        ↓
Model Training & Selection (XGBoost wins)
        ↓
Serialized Artifacts (model.pkl, encoder.pkl)
        ↓
Streamlit Web App → Real-time Predictions
```

## 📊 Model Performance

| Model | RMSE | R² Score |
|---|---|---|
| **XGBoost (tuned)** | **0.134** | **0.947** |
| XGBoost (default) | 0.135 | 0.946 |
| Random Forest (tuned) | 0.147 | 0.936 |
| Random Forest (default) | 0.146 | 0.938 |
| Ridge Regression | 0.160 | 0.925 |
| Linear Regression | 0.160 | 0.925 |
| Decision Tree | 0.181 | 0.904 |
| Lasso Regression | 0.223 | 0.854 |

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn, XGBoost
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit
- **Model Persistence:** joblib
- **Environment:** VS Code, Jupyter Notebook

## 📁 Project Structure

```
Car_Price_Prediction/
│
├── app.py                     # Streamlit web application
├── car_prediction.ipynb       # Full notebook: EDA, cleaning, model training
├── cardekho_dataset.csv       # Raw dataset
├── clean_car_data.csv         # Cleaned, feature-engineered dataset
├── model.pkl                  # Trained XGBoost model
├── encoder.pkl                # Fitted OneHotEncoder
├── feature_columns.pkl        # Expected model input column order
├── categorical_options.pkl    # Dropdown options for the Streamlit UI
├── requirements.txt           # Python dependencies
└── README.md
```

## ⚙️ How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/car-price-prediction.git
   cd car-price-prediction
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in your terminal (usually `http://localhost:8501`)

## 🖼️ Features

- **Real-time price prediction** based on brand, model, age, mileage, engine, and more
- **Brand-filtered model dropdown** — only shows models matching the selected brand
- **Exploratory Data Analysis tab** — price distribution, fuel type breakdown, brand-wise averages, correlation heatmap
- Dark, dashboard-style UI

## 🔮 Future Scope

- Add SHAP-based explainability to show *why* a price was predicted
- Integrate real-time market data via API for dynamic pricing
- Add location-based price adjustments (prices vary by city/region in India)
- Multilingual support (Hindi/English)
- Migrate to a more scalable deployment (Docker + cloud hosting)

## 📚 Dataset Source

[CarDekho Used Car Data](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data) — Kaggle, by manishkr1754


*Built as part of placement preparation, focused on solving real-world, India-relevant problems through machine learning.*
