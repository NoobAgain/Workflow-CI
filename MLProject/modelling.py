import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Load data
df = pd.read_csv('anime_preprocessing.csv')
X = df[['Score', 'Votes', 'Favourites_count']]
y = df['Popularity_ranking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training dengan MLflow
with mlflow.start_run() as run:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("n_estimators", 100)
    mlflow.sklearn.log_model(model, "model")
    
    joblib.dump(model, "model.joblib")
    
    print(f"Run ID: {run.info.run_id}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    print("✅ Selesai! Buka http://127.0.0.1:5000")
