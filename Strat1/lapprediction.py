"""
F1 Lap Time Prediction Model V3

This module builds and trains the machine learning model to predict F1 lap times.
"""

import pickle
import fastf1
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
from joblib import dump, load
import warnings
from typing import Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os

from utils.encoder import MultiHotEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Enable F1 data caching for faster subsequent loads
fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')


class EnhancedFeatureEngineer:
    """Advanced feature engineering for F1 lap time prediction."""
    
    def __init__(self):
        self.compound_degradation_rates = {
            'SOFT': 0.05,
            'MEDIUM': 0.03,
            'HARD': 0.02,
            'INTERMEDIATE': 0.02,
            'WET': 0.01
        }
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for better prediction accuracy.
        
        Args:
            data: Raw F1 data
            
        Returns:
            Enhanced DataFrame with new features
        """
        data = data.copy()
        
        # Tire degradation features
        if 'Compound' in data.columns and 'TyreLife' in data.columns:
            data['TireDegradation'] = data.apply(
                lambda row: self._calculate_tire_degradation(row['Compound'], row['TyreLife']), 
                axis=1
            )
        
        # Fuel load effect (approximate)
        if 'LapNumber' in data.columns:
            # Group by session to get proper total laps per session
            data['FuelLoad'] = data.groupby(['EventName', 'EventYear'])['LapNumber'].transform(
                lambda x: (x.max() - x + 1) * 1.6
            )
            data['FuelEffect'] = data['FuelLoad'] * 0.035  # seconds per kg
        
        # Track evolution (grip improvement over session)
        if 'LapNumber' in data.columns:
            data['TrackEvolution'] = np.log1p(data['LapNumber']) * 0.1
        
        # Temperature differential
        if all(col in data.columns for col in ['AirTemp', 'TrackTemp']):
            data['TempDiff'] = data['TrackTemp'] - data['AirTemp']
            data['TempRatio'] = data['TrackTemp'] / (data['AirTemp'] + 1)
        
        # Qualifying pace differential
        if 'LapTime_Qualifying' in data.columns:
            data['QualifyingGap'] = data.groupby(['EventName', 'EventYear'])['LapTime_Qualifying'].transform(
                lambda x: x - x.min()
            )
        
        # Stint characteristics
        data = self._add_stint_features(data)
        
        # Position-based features
        if 'Position' in data.columns:
            data['PositionGroup'] = pd.cut(data['Position'], 
                                         bins=[0, 3, 6, 10, 20], 
                                         labels=['Top3', 'Top6', 'Midfield', 'Back'])
        
        # Weather condition features
        if 'Rainfall' in data.columns:
            data['WeatherCondition'] = data['Rainfall'].map({
                True: 'Wet', 
                False: 'Dry'
            })
        
        # Team performance tiers
        data = self._add_team_tiers(data)
        
        return data
    
    def _calculate_tire_degradation(self, compound: str, tire_life: int) -> float:
        """Calculate tire degradation effect."""
        base_rate = self.compound_degradation_rates.get(compound, 0.03)
        return base_rate * tire_life
    
    def _add_stint_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add stint-related features."""
        if 'FreshTyre' in data.columns and 'Driver' in data.columns:
            try:
                # Create stint groups more safely
                data = data.sort_values(['Driver', 'EventName', 'EventYear', 'LapNumber'])
                
                # Create stint identifier
                data['StintID'] = (
                    data.groupby(['Driver', 'EventName', 'EventYear'])['FreshTyre']
                    .cumsum()
                )
                
                # Calculate stint position and length
                stint_stats = (
                    data.groupby(['Driver', 'EventName', 'EventYear', 'StintID'])
                    .agg({
                        'LapNumber': ['count', 'min']
                    })
                )
                stint_stats.columns = ['StintLength', 'StintStartLap']
                stint_stats = stint_stats.reset_index()
                
                # Merge back to main data
                data = data.merge(
                    stint_stats,
                    on=['Driver', 'EventName', 'EventYear', 'StintID'],
                    how='left'
                )
                
                # Calculate stint position
                data['StintPosition'] = (
                    data.groupby(['Driver', 'EventName', 'EventYear', 'StintID'])
                    .cumcount() + 1
                )
                
                # Calculate stint progress
                data['StintProgress'] = data['StintPosition'] / data['StintLength']
                
                # Clean up temporary column
                data = data.drop('StintID', axis=1)
                
            except Exception as e:
                logger.warning(f"Error creating stint features: {e}")
                # Create default values
                data['StintPosition'] = 1
                data['StintLength'] = 20
                data['StintProgress'] = 0.5
        
        return data
    
    def _add_team_tiers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add team performance tiers based on qualifying times."""
        if 'Team' in data.columns and 'LapTime_Qualifying' in data.columns:
            try:
                team_performance = data.groupby('Team')['LapTime_Qualifying'].median().sort_values()
                
                # Divide teams into tiers
                n_teams = len(team_performance)
                if n_teams >= 3:
                    tier1_teams = team_performance.index[:n_teams//3]
                    tier2_teams = team_performance.index[n_teams//3:2*n_teams//3]
                    tier3_teams = team_performance.index[2*n_teams//3:]
                    
                    def get_team_tier(team):
                        if team in tier1_teams:
                            return 'Tier1'
                        elif team in tier2_teams:
                            return 'Tier2'
                        else:
                            return 'Tier3'
                    
                    data['TeamTier'] = data['Team'].apply(get_team_tier)
                else:
                    data['TeamTier'] = 'Tier2'  # Default if not enough teams
                    
            except Exception as e:
                logger.warning(f"Error creating team tiers: {e}")
                data['TeamTier'] = 'Tier2'  # Default value
        
        return data


class ModelOptimizer:
    """Advanced model optimization using Optuna."""
    
    def __init__(self):
        self.best_params = {}
        self.best_score = float('inf')
    
    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series, 
                        n_trials: int = 50) -> Dict[str, Any]:  # Reduced trials for faster execution
        """
        Optimize XGBoost hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
                'random_state': 100,
                'n_jobs': -1
            }
            
            # Create preprocessor
            preprocessor = self._build_preprocessor(X_train.columns.tolist())
            
            # Create and train model
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(**params))
            ])
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            return mean_squared_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best XGBoost parameters: {self.best_params}")
        logger.info(f"Best validation MSE: {self.best_score:.4f}")
        
        return self.best_params
    
    def _build_preprocessor(self, available_columns: list):
        """Build the preprocessing pipeline based on available columns."""
        # Base categorical features that should always be present
        base_categorical = ['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear', 'Rainfall']
        
        # Enhanced categorical features
        enhanced_categorical = ['WeatherCondition', 'PositionGroup', 'TeamTier']
        
        # Status features
        status_features = ['TrackStatus']
        
        # Base numerical features
        base_numerical = ['LapTime_Qualifying', 'AirTemp', 'TrackTemp', 'TyreLife', 'LapNumber', 'StartingPosition', 'Position']
        
        # Enhanced numerical features
        enhanced_numerical = [
            'TireDegradation', 'FuelLoad', 'FuelEffect', 'TrackEvolution', 
            'TempDiff', 'TempRatio', 'QualifyingGap', 'StintPosition', 
            'StintLength', 'StintProgress'
        ]
        
        # Filter features based on what's actually available
        categorical_features = [col for col in base_categorical + enhanced_categorical if col in available_columns]
        numerical_features = [col for col in base_numerical + enhanced_numerical if col in available_columns]
        status_features = [col for col in status_features if col in available_columns]
        
        transformers = []
        
        if categorical_features:
            transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features))
        
        if status_features:
            transformers.append(('status', MultiHotEncoder(), status_features))
        
        if numerical_features:
            transformers.append(('numerical', RobustScaler(), numerical_features))
        
        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )


class EnhancedModelPipeline:
    """Enhanced model pipeline with multiple algorithms and validation."""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = EnhancedFeatureEngineer()
        self.optimizer = ModelOptimizer()
        self.feature_importance = {}
        
    def build_model_pipeline(self, model_type: str = 'xgboost', optimize: bool = True, available_columns: list = None) -> Pipeline:
        """
        Create an enhanced model pipeline with advanced preprocessing.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'gradient_boosting')
            optimize: Whether to optimize hyperparameters
            available_columns: List of available column names
            
        Returns:
            Complete ML pipeline
        """
        if available_columns is None:
            available_columns = []
        
        # Create preprocessing pipeline based on available columns
        preprocessor = self.optimizer._build_preprocessor(available_columns)
        
        # Select and configure model
        if model_type == 'xgboost':
            if optimize and hasattr(self.optimizer, 'best_params') and self.optimizer.best_params:
                # Use optimized parameters
                params = self.optimizer.best_params.copy()
                params.update({
                    'objective': 'reg:squarederror',
                    'random_state': 100,
                    'n_jobs': -1
                })
            else:
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 1000,
                    'learning_rate': 0.1,
                    'max_depth': 7,
                    'random_state': 100,
                    'n_jobs': -1
                }
            regressor = XGBRegressor(**params)
            
        elif model_type == 'lightgbm':
            regressor = LGBMRegressor(
                objective='regression',
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=7,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=100,
                n_jobs=-1,
                verbose=-1
            )
            
        elif model_type == 'random_forest':
            regressor = RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=100,
                n_jobs=-1
            )
            
        elif model_type == 'gradient_boosting':
            regressor = GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=6,
                random_state=100
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Combine preprocessing and model
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        return model_pipeline
    
    def load_and_prepare_data(self, enhance_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare F1 data with enhanced feature engineering.
        
        Args:
            enhance_features: Whether to apply advanced feature engineering
            
        Returns:
            Tuple of (features, target)
        """
        # Load the training data
        try:
            with open('./utils/f1_data.pkl', 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            logger.error("F1 data file not found. Please ensure f1_data.pkl exists in utils/")
            raise
        
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Data columns: {list(data.columns)}")
        
        # Apply feature engineering
        if enhance_features:
            logger.info("Applying advanced feature engineering...")
            try:
                data = self.feature_engineer.create_advanced_features(data)
                logger.info("Feature engineering completed successfully")
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")
                logger.info("Continuing with base features only...")
        
        # Select required features
        base_columns = [
            'Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 
            'EventYear', 'Rainfall', 'TrackStatus', 'LapTime_Qualifying', 
            'AirTemp', 'TrackTemp', 'TyreLife', 'LapNumber', 'LapTime', 
            'StartingPosition', 'Position'
        ]
        
        # Add enhanced features if available
        enhanced_columns = [
            'TireDegradation', 'FuelLoad', 'FuelEffect', 'TrackEvolution',
            'TempDiff', 'TempRatio', 'QualifyingGap', 'StintPosition',
            'StintLength', 'StintProgress', 'WeatherCondition', 'PositionGroup', 'TeamTier'
        ]
        
        available_columns = [col for col in base_columns + enhanced_columns if col in data.columns]
        
        # Check for missing base columns
        missing_columns = [col for col in base_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Remove missing columns from base_columns
            base_columns = [col for col in base_columns if col in data.columns]
            available_columns = base_columns + [col for col in enhanced_columns if col in data.columns]
        
        data = data[available_columns]
        logger.info(f"Data shape after column selection: {data.shape}")
        logger.info(f"Selected columns: {available_columns}")
        
        # Clean data
        initial_rows = len(data)
        data = data.dropna()
        logger.info(f"Removed {initial_rows - len(data)} rows with missing values")
        
        # Remove outliers (lap times > 2 minutes or < 30 seconds)
        if 'LapTime' in data.columns:
            initial_rows = len(data)
            data = data[(data['LapTime'] >= 30) & (data['LapTime'] <= 120)]
            logger.info(f"Removed {initial_rows - len(data)} outlier lap times")
        
        # Separate features and target
        X = data.drop(columns=['LapTime'])
        y = data['LapTime']
        
        logger.info(f"Final features shape: {X.shape}")
        logger.info(f"Final target shape: {y.shape}")
        
        return X, y
    
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics
    
    def plot_feature_importance(self, model: Pipeline, feature_names: list, top_n: int = 20):
        """Plot feature importance."""
        try:
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                importances = model.named_steps['regressor'].feature_importances_
                
                # Get feature names after preprocessing
                try:
                    feature_names_out = model.named_steps['preprocessor'].get_feature_names_out()
                except:
                    feature_names_out = [f'feature_{i}' for i in range(len(importances))]
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names_out,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
                
                # Plot
                plt.figure(figsize=(10, 8))
                sns.barplot(data=importance_df, y='feature', x='importance')
                plt.title('Top Feature Importances')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.show()
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")
    
    def train_and_save_enhanced_model(self, model_type: str = 'xgboost', 
                                    optimize: bool = False, 
                                    save_path: str = 'utils/lapprediction_model_enhanced.joblib'):
        """
        Train and save the enhanced model with comprehensive evaluation.
        
        Args:
            model_type: Type of model to train
            optimize: Whether to optimize hyperparameters
            save_path: Path to save the trained model
        """
        logger.info("Starting enhanced model training...")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data(enhance_features=True)
        
        # Split data - use a simpler strategy if EventName stratification fails
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=X['EventName']
            )
        except:
            logger.warning("Stratified split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Optimize hyperparameters if requested
        if optimize and model_type == 'xgboost':
            logger.info("Optimizing hyperparameters...")
            try:
                self.optimizer.optimize_xgboost(X_train_split, y_train_split, X_val, y_val)
            except Exception as e:
                logger.warning(f"Optimization failed: {e}, continuing with default parameters")
        
        # Build and train model
        logger.info(f"Building {model_type} model pipeline...")
        model = self.build_model_pipeline(model_type, optimize, X_train.columns.tolist())
        
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        train_metrics = self.evaluate_model(model, X_train, y_train)
        test_metrics = self.evaluate_model(model, X_test, y_test)
        
        logger.info("Training Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=3,  
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            logger.info(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std() * 2):.4f})")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
        
        # Feature importance
        self.plot_feature_importance(model, X.columns.tolist())
        
        # Save model
        logger.info(f"Saving model to {save_path}...")
        dump(model, save_path)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_count': X.shape[1],
            'training_samples': len(X_train)
        }
        
        try:
            with open(save_path.replace('.joblib', '_metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
        
        logger.info("Enhanced model training completed successfully!")
        
        return model, test_metrics


def train_and_save_model():
    """
    Legacy function for backward compatibility.
    """
    pipeline = EnhancedModelPipeline()
    model, metrics = pipeline.train_and_save_enhanced_model(
        model_type='xgboost', 
        optimize=False, 
        save_path='utils/lapprediction_model.joblib'
    )
    return model, metrics


if __name__ == "__main__":
    # Train enhanced model with optimization
    pipeline = EnhancedModelPipeline()
    
    # Train multiple models for comparison
    models_to_train = ['xgboost', 'lightgbm', 'random_forest']
    results = {}
    
    for model_type in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"{'='*50}")
        
        try:
            model, metrics = pipeline.train_and_save_enhanced_model(
                model_type=model_type,
                optimize=(model_type == 'xgboost'),  #optimize XGBoost
                save_path=f'utils/lapprediction_model_{model_type}.joblib'
            )
            results[model_type] = metrics
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    if results:
        logger.info(f"\n{'='*50}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*50}")
        
        comparison_df = pd.DataFrame(results).T
        logger.info(comparison_df.round(4))
        
        # Find best model
        best_model = comparison_df['rmse'].idxmin()
        logger.info(f"\nBest model: {best_model.upper()}")
        
        # Copy best model to default location
        import shutil
        best_model_path = f'utils/lapprediction_model_{best_model}.joblib'
        default_path = 'utils/lapprediction_model.joblib'
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, default_path)
            logger.info(f"Copied best model ({best_model}) to {default_path}")