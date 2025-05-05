import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # Check if the output directory exists; if not, create it
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the data from the input CSV file
    df = pd.read_csv(input_path, parse_dates=["Date"])

    # Lag Features
    df['Revenue_Y-1'] = df['Revenue'].shift(12)
    df['Exchange_Rate_Y-1'] = df['Exchange_Rate_NGN_USD'].shift(12)
    df['Inflation_Rate_Y-1'] = df['Inflation_Rate_Percent'].shift(12)

    # Revenue Growth % compared to same month previous year
    df['Revenue_Growth(%)'] = ((df['Revenue'] - df['Revenue_Y-1']) / df['Revenue_Y-1']) * 100

    # Dealing with date columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    # Drop the original Date column
    df = df.drop(columns=['Date'])

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Save the cleaned data to the specified output path
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Data saved to {output_path}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data("./Revenue_Forecast/Data/revenue_forecasting_monthly_dummy_data.csv", "processed_data/revenue_forecast_cleaned.csv")
