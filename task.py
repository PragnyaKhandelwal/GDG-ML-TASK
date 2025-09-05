# Section 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from google.colab import drive

# Section 2: Load Data
weather_plant1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/archive/Plant_1_Weather_Sensor_Data.csv')
generation_plant1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/archive/Plant_1_Generation_Data.csv')

print("Columns in weather_plant1:", weather_plant1.columns)
print("Columns in generation_plant1:", generation_plant1.columns)

df_plant1 = pd.merge(weather_plant1, generation_plant1, on='DATE_TIME')

weather_plant2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/archive/Plant_2_Weather_Sensor_Data.csv')
generation_plant2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/archive/Plant_2_Generation_Data.csv')

print("Columns in weather_plant2:", weather_plant2.columns)
print("Columns in generation_plant2:", generation_plant2.columns)

df_plant2 = pd.merge(weather_plant2, generation_plant2, on='DATE_TIME')

# Optional: Combine data from both plants
df = pd.concat([df_plant1, df_plant2], ignore_index=True)

# Section 3: EDA
print(df.head())
print(df.info())
print(df.describe())

df.hist(bins=30, figsize=(12, 8))
plt.show()

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()

# Section 4: Preprocessing
df = df.dropna()  # Drop rows with missing values

# Section 5: Train/Test Split
# Replace these with actual weather feature columns from your dataset
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
X = df[features]

# Replace with actual generation/target column name
y = df['DAILY_YIELD']

# Shuffle=False because data is time series
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# Section 6: Modeling
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=0),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Fixed RMSE calculation without squared param
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    plt.figure()
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(name)
    plt.legend()
    plt.show()

    # Section 7: Model Comparison Table
results_df = pd.DataFrame(results)
print(results_df)

# Section 8: Final Prediction for Next Days
best_model = models['RandomForest']  # Set best model here
latest_data = X.tail(2)  # Last 2 rows of features for prediction
future_predictions = best_model.predict(latest_data)
print("Predictions for next 2 days:", future_predictions)
