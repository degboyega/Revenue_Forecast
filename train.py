import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(input_path, model_output_path):
    # Load preprocessed data
    df = pd.read_csv(input_path)

    # Prepare feature matrix and target vector
    X = df.drop(columns=['Revenue']) 
    y = df['Revenue']  # target column

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model training complete. Model saved to {model_output_path}")


# Run the training function
if __name__ == "__main__":
    train_model("Revenue_Forecast/processed_data/revenue_forecast_cleaned.csv", "models/revenue_forecasting_rf.pkl")


