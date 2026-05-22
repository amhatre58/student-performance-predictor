


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# 1. Load the Dataset
df = pd.read_csv('student_data.csv')
# --- DATA CLEANING ---
df = df.dropna()
# 2. Exploratory Data Analysis & Preprocessing
print("Dataset Preview:")
print(df.head())
# Handling Categorical variables (e.g., Extracurricular Activities: Yes/No)
le = LabelEncoder()
for col in df.select_dtypes(include=['str']).columns:
    df[col] = le.fit_transform(df[col])
# 3. Feature Selection
# Assuming 'G3' is the target variable
X = df.drop('G3', axis=1) 
y = df['G3']
# 4. Data Splitting & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_cols=['studytime','absences','G2']
scaler = StandardScaler()
X_train_scaled=X_train.copy()
X_test_scaled=X_test.copy()
X_train_scaled[num_cols]= scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
# 5. Model Building
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
# 6. Evaluation
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    results.append({"Model": name, "RMSE": rmse, "R2 Score": r2})
# Display Results
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df)
# 7. Visualization of Results (Actual vs Predicted)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions, alpha=0.5, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Student Performance')
plt.show()
# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()
