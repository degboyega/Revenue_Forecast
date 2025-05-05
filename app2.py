import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Revenue Forecasting System", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stSelectbox>div>div>div>div {
        color: #333;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("💰 Revenue Forecasting System")
st.markdown("""
This application helps forecast future revenue based on historical data, exchange rates (NGN/USD), 
and inflation rates. Upload your data, train the model, and generate forecasts.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
                           ["Data Upload & Preprocessing", 
                            "Model Training", 
                            "Generate Forecasts",
                            "About"])

if app_mode == "Data Upload & Preprocessing":
    st.header("📊 Data Upload & Preprocessing")
    
    # File upload section
    st.subheader("Upload Historical Data")
    uploaded_file = st.file_uploader("Choose a CSV file with columns: Date, Revenue, Exchange_Rate_NGN_USD, Inflation_Rate_Percent", 
                                    type=["csv"])
    
    if uploaded_file is not None:
        # Load and display data
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
        
        st.success("Data successfully loaded!")
        st.subheader("Preview of Uploaded Data")
        st.write(df.head())
        
        # Data statistics
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        # Data visualization
        st.subheader("Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Revenue Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df, x="Date", y="Revenue", ax=ax)
            ax.set_title("Revenue Trend")
            st.pyplot(fig)
            
        with col2:
            st.write("Exchange Rate (NGN/USD) Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df, x="Date", y="Exchange_Rate_NGN_USD", ax=ax)
            ax.set_title("Exchange Rate Trend")
            st.pyplot(fig)
        
        # Preprocessing button
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Create processed_data directory if it doesn't exist
                    os.makedirs("processed_data", exist_ok=True)
                    
                    # Perform preprocessing steps
                    df['Revenue_Y-1'] = df['Revenue'].shift(12)
                    df['Exchange_Rate_Y-1'] = df['Exchange_Rate_NGN_USD'].shift(12)
                    df['Inflation_Rate_Y-1'] = df['Inflation_Rate_Percent'].shift(12)
                    df['Revenue_Growth(%)'] = ((df['Revenue'] - df['Revenue_Y-1']) / df['Revenue_Y-1']) * 100
                    df['Year'] = df['Date'].dt.year
                    df['Month'] = df['Date'].dt.month
                    df['Quarter'] = df['Date'].dt.quarter
                    df = df.drop(columns=['Date']).dropna().reset_index(drop=True)
                    
                    # Save processed data
                    processed_path = "processed_data/revenue_forecast_cleaned.csv"
                    df.to_csv(processed_path, index=False)
                    
                    st.success(f"Preprocessing complete! Data saved to {processed_path}")
                    st.subheader("Preview of Processed Data")
                    st.write(df.head())
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")

elif app_mode == "Model Training":
    st.header("🤖 Model Training")
    
    # Check if processed data exists
    if not os.path.exists("processed_data/revenue_forecast_cleaned.csv"):
        st.warning("Please upload and preprocess your data first in the 'Data Upload & Preprocessing' section.")
    else:
        # Load processed data
        df = pd.read_csv("processed_data/revenue_forecast_cleaned.csv")
        
        st.subheader("Processed Data Preview")
        st.write(df.head())
        
        # Model training parameters
        st.subheader("Model Training Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Size Percentage", 10, 40, 20) / 100
            random_state = st.number_input("Random State", value=42)
            
        with col2:
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=100)
            max_depth = st.number_input("Max Depth", min_value=1, max_value=30, value=None)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import train_test_split
                    
                    # Prepare data
                    X = df.drop(columns=['Revenue'])
                    y = df['Revenue']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state)
                    
                    # Initialize and train model
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Save model
                    os.makedirs("models", exist_ok=True)
                    model_path = "models/revenue_forecasting_rf.pkl"
                    joblib.dump(model, model_path)
                    
                    st.success(f"Model trained successfully! Saved to {model_path}")
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Squared Error", f"{mse:,.2f}")
                    
                    with col2:
                        st.metric("Root Mean Squared Error", f"{rmse:,.2f}")
                    
                    with col3:
                        st.metric("R-squared Score", f"{r2:.4f}")
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

elif app_mode == "Generate Forecasts":
    st.header("🔮 Generate Forecasts")
    
    # Check if model exists
    if not os.path.exists("models/revenue_forecasting_rf.pkl"):
        st.warning("Please train the model first in the 'Model Training' section.")
    else:
        # Load model
        model = joblib.load("models/revenue_forecasting_rf.pkl")
        
        # Forecast parameters
        st.subheader("Forecast Parameters")
        
        # Get the most recent data point for lag features
        df = pd.read_csv("processed_data/revenue_forecast_cleaned.csv")
        last_row = df.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_months = st.number_input("Number of Months to Forecast", min_value=1, max_value=36, value=12)
            initial_revenue = st.number_input("Initial Revenue (for growth calculation)", value=last_row['Revenue'])
            
        with col2:
            initial_exchange_rate = st.number_input("Initial Exchange Rate (NGN/USD)", value=last_row['Exchange_Rate_NGN_USD'])
            initial_inflation_rate = st.number_input("Initial Inflation Rate (%)", value=last_row['Inflation_Rate_Percent'])
        
        # Future values input
        st.subheader("Future Economic Indicators")
        st.write("Provide estimates for future exchange rates and inflation rates.")
        
        # Default values for future economic indicators
        default_exchange = [initial_exchange_rate * (1 + 0.01*i) for i in range(forecast_months)]
        default_inflation = [initial_inflation_rate * (1 + 0.005*i) for i in range(forecast_months)]
        
        future_data = []
        for i in range(forecast_months):
            col1, col2 = st.columns(2)
            month = (datetime.now() + timedelta(days=30*i)).strftime("%Y-%m")
            
            with col1:
                exchange = st.number_input(f"Exchange Rate (NGN/USD) for {month}", 
                                         value=default_exchange[i], 
                                         key=f"exchange_{i}")
            
            with col2:
                inflation = st.number_input(f"Inflation Rate (%) for {month}", 
                                           value=default_inflation[i], 
                                           key=f"inflation_{i}")
            
            future_data.append({
                'Year': (datetime.now() + timedelta(days=30*i)).year,
                'Month': (datetime.now() + timedelta(days=30*i)).month,
                'Quarter': (datetime.now() + timedelta(days=30*i)).month // 4 + 1,
                'Exchange_Rate_NGN_USD': exchange,
                'Inflation_Rate_Percent': inflation
            })
        
        # Generate forecast button
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Prepare forecast dataframe with ALL expected features
                    forecast_df = pd.DataFrame(future_data)
                    
                    # Initialize with the last known values
                    last_revenue = initial_revenue
                    last_exchange = initial_exchange_rate
                    last_inflation = initial_inflation_rate
                    
                    predicted_revenues = []
                    
                    for i in range(forecast_months):
                        # Create feature dictionary with ALL features the model expects
                        features = {
                            'Year': forecast_df.at[i, 'Year'],
                            'Month': forecast_df.at[i, 'Month'],
                            'Quarter': forecast_df.at[i, 'Quarter'],
                            'Exchange_Rate_NGN_USD': forecast_df.at[i, 'Exchange_Rate_NGN_USD'],
                            'Inflation_Rate_Percent': forecast_df.at[i, 'Inflation_Rate_Percent'],
                            'Revenue_Y-1': last_revenue,
                            'Exchange_Rate_Y-1': last_exchange,
                            'Inflation_Rate_Y-1': last_inflation,
                            'Revenue_Growth(%)': 0
                        }
                        
                        # Predict revenue
                        predicted_revenue = model.predict([list(features.values())])[0]
                        
                        predicted_revenues.append(predicted_revenue)
                        
                        # Update last values for the next prediction
                        last_revenue = predicted_revenue
                        last_exchange = forecast_df.at[i, 'Exchange_Rate_NGN_USD']
                        last_inflation = forecast_df.at[i, 'Inflation_Rate_Percent']
                    
                    forecast_df['Predicted_Revenue'] = predicted_revenues
                    
                    # Display forecast
                    st.subheader("Revenue Forecasts")
                    st.write(forecast_df[['Year', 'Month', 'Predicted_Revenue']])
                    
                except Exception as e:
                    st.error(f"Error during forecast generation: {str(e)}")

elif app_mode == "About":
    st.header("📚 About")
    st.markdown("""
    This app uses machine learning (Random Forest Regressor) to predict future revenue 
    based on historical data, exchange rates, and inflation rates.
    It allows you to upload data, train the model, and generate forecasts.
    
    - **Data:** You need a CSV file containing the columns `Date`, `Revenue`, `Exchange_Rate_NGN_USD`, `Inflation_Rate_Percent`.
    - **Model:** Random Forest Regressor is used for revenue prediction.
    - **Forecasting:** Forecasts can be generated for up to 36 months ahead.
    """)
