# Student Performance Predictor

A machine learning pipeline built with Python to predict final student grades (`G3`) using demographic, behavioral, and academic features. 

## 📊 Project Overview
This project cleans and preprocesses student academic data, encodes categorical variables, and scales numerical features. It trains and compares two distinct machine learning models to see which one performs better.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn

## 🚀 Key Features
* **Automated Data Cleaning:** Handles missing data and automatically encodes categorical variables.
* **Feature Scaling:** Uses `StandardScaler` on key numerical columns (`studytime`, `absences`, `G2`).
* **Model Comparison:** Evaluates and compares **Linear Regression** and **Random Forest Regressor**.
* **Visual Diagnostics:** Generates a correlation heatmap and an Actual vs. Predicted scatter plot.

## 📈 Code Structure
* `predict.py` - The main Python script containing the pipeline.
* `student_data.csv` - The source dataset containing student attributes (Required to run locally).

## ⚙️ How to Run Locally
1. Clone this repository or download `predict.py`.
2. Ensure you have your dataset named `student_data.csv` in the same directory.
3. Run the script using your terminal or IDE:
   ```bash
   python predict.py
   ```
