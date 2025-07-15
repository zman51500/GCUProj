"""
Simple Driver MSE Analysis

Calculate Mean Squared Error for each driver in the dataset.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import feature engineering
from lapprediction import EnhancedFeatureEngineer

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_driver_mse():
    """Calculate MSE for each driver in the dataset."""
    
    # Load model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'lapprediction_model.joblib')
        model = load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load data
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'f1_data.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Data loaded successfully: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Clean data
    data = data.dropna()
    data = data[(data['LapTime'] >= 30) & (data['LapTime'] <= 120)]
    print(f"📊 Cleaned data shape: {data.shape}")
    
    # Reset index after cleaning to ensure alignment
    data = data.reset_index(drop=True)
    
    # Apply feature engineering
    try:
        engineer = EnhancedFeatureEngineer()
        data_enhanced = engineer.create_advanced_features(data)
        print("✅ Feature engineering applied")
        
        # Ensure indices are aligned after feature engineering
        data_enhanced = data_enhanced.reset_index(drop=True)
        
    except Exception as e:
        print(f"⚠️ Feature engineering failed, using original data: {e}")
        data_enhanced = data.reset_index(drop=True)
    
    # Prepare features and target
    X = data_enhanced.drop('LapTime', axis=1)
    y = data_enhanced['LapTime']
    
    # Make predictions
    try:
        predictions = model.predict(X)
        print("✅ Predictions generated")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return
    
    # Calculate MSE for each driver
    driver_mse = {}
    
    # Use the enhanced data for driver identification since indices are now aligned
    for driver in data_enhanced['Driver'].unique():
        driver_mask = data_enhanced['Driver'] == driver
        driver_actual = y[driver_mask]
        driver_pred = predictions[driver_mask]
        
        if len(driver_actual) > 0:
            mse = mean_squared_error(driver_actual, driver_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(driver_actual - driver_pred))
            
            driver_mse[driver] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Count': len(driver_actual),
                'Avg_Actual': driver_actual.mean(),
                'Avg_Predicted': driver_pred.mean()
            }
    
    # Create results dataframe
    results_df = pd.DataFrame(driver_mse).T
    results_df = results_df.sort_values('MSE')
    
    # Display results
    print("\n" + "=" * 80)
    print("DRIVER MSE ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 Results for {len(results_df)} drivers:")
    print(results_df.round(4))
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"  Best MSE (lowest): {results_df['MSE'].min():.4f} ({results_df['MSE'].idxmin()})")
    print(f"  Worst MSE (highest): {results_df['MSE'].max():.4f} ({results_df['MSE'].idxmax()})")
    print(f"  Average MSE: {results_df['MSE'].mean():.4f}")
    print(f"  Median MSE: {results_df['MSE'].median():.4f}")
    print(f"  MSE Standard Deviation: {results_df['MSE'].std():.4f}")
    
    print(f"\n⏱️ RMSE Summary:")
    print(f"  Best RMSE: {results_df['RMSE'].min():.4f}s ({results_df['RMSE'].idxmin()})")
    print(f"  Worst RMSE: {results_df['RMSE'].max():.4f}s ({results_df['RMSE'].idxmax()})")
    print(f"  Average RMSE: {results_df['RMSE'].mean():.4f}s")
    
    # Save results
    try:
        output_path = os.path.join(os.path.dirname(__file__), 'driver_mse_results.csv')
        results_df.to_csv(output_path)
        print(f"\n💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    
    return results_df

if __name__ == "__main__":
    results = calculate_driver_mse()