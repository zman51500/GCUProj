# 🏁 Strat1 - F1 Strategy Prediction System

An advanced Formula 1 tire strategy optimization and race analysis platform built with machine learning and real-time F1 data.

![F1 Strategy Prediction](https://img.shields.io/badge/F1-Strategy%20Prediction-red?style=for-the-badge&logo=formula1)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green?style=for-the-badge&logo=streamlit)
![FastF1](https://img.shields.io/badge/FastF1-3.5.3-orange?style=for-the-badge)

## 🌟 Features

### 🎯 Strategy Optimization
- **AI-Powered Predictions**: Advanced XGBoost model trained on historical F1 data
- **Real-Time Strategy Generation**: Generate optimal tire strategies for any race scenario
- **Customizable Parameters**: Adjust pit stop times, weather conditions, and race parameters
- **Multiple Strategy Options**: Compare top strategies with detailed analysis

### 📊 Race Analysis
- **2025 Season Data**: Complete race analysis for finished F1 races
- **Driver Performance**: MSE analysis and lap time predictions per driver
- **Interactive Visualizations**: Position progression, lap time analysis, and performance distributions
- **Pit Stop Analysis**: Detailed tire strategy breakdowns and pit stop timing

### 🔧 Advanced Features
- **Enhanced Feature Engineering**: 20+ engineered features including tire degradation, fuel effects, and track evolution
- **Multi-Model Support**: XGBoost, LightGBM, and Random Forest algorithms
- **Hyperparameter Optimization**: Optuna-based automatic parameter tuning
- **Adjustable Pit Stop Times**: Realistic pit stop penalty modeling (18-30 seconds)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd GCUProj/V3
```

2. **Install dependencies**
```bash
pip3 install -r requirements.txt
```

3. **Set up F1 data cache** (optional but recommended)
```python
# The system will automatically create a cache directory
# For faster data loading, ensure you have sufficient disk space
```

### Running the Application

1. **Start the Streamlit app**
```bash
streamlit run Strat_Prediction.py
```

2. **Access the application**
   - Open your browser to `https://zman51500-gcuproj-strat1strat-prediction-bu8jhu.streamlit.app/`
   - The main strategy prediction interface will load

3. **Navigate between pages**
   - **Strategy Prediction**: Main optimization interface
   - **2025 Race Data**: Historical race analysis and visualization

## 📁 Project Structure

```
Strat1/
├── Strat_Prediction.py          # Main Streamlit application
├── lapprediction.py             # ML model training and feature engineering
├── rec.py                       # Strategy recommendation engine
├── requirements.txt             # Python dependencies
├── pages/                       # Streamlit pages
│   └── 1_2025_Race_Data.py     # Race analysis page
└── utils/                       # Utility modules
    ├── encoder.py               # Custom data encoders
    ├── f1_data.pkl             # F1 training data (generated)
    ├── lapprediction_model.joblib # Trained ML model (generated)
    └── test.py                  # Model testing and validation
```

## 🛠️ Usage Guide

### Strategy Prediction

1. **Configure Race Parameters**
   - Select race track, qualifying time, starting position
   - Set total race laps and environmental conditions
   - Choose team and driver

2. **Adjust Strategy Settings**
   - Number of tire stints (2-5)
   - Pit stop penalty time (18-30 seconds)
   - Rain conditions toggle

3. **Generate Optimal Strategy**
   - Click "🔍 Recommend Optimal Strategy"
   - View top 5 strategies with detailed breakdowns
   - Analyze lap-by-lap predictions

4. **Manual Strategy Input**
   - Create custom strategies with manual tire compound selection
   - View predicted lap times and total race time
   - Compare against AI recommendations

### Race Analysis

1. **Select Completed Race**
   - Choose from 2025 F1 season races
   - View race results and statistics

2. **Analyze Performance**
   - Position progression throughout the race
   - Lap time analysis with tire compound visualization
   - Performance distribution comparisons

3. **Driver Comparison**
   - Multi-select drivers for detailed comparison
   - Pit stop timing and strategy analysis

## 🤖 Machine Learning Pipeline

### Model Training
```bash
# Train new model with latest data
python3 lapprediction.py
```

### Model Testing
```bash
# Run comprehensive model validation
python3 utils/test.py
```

### Features Used
- **Driver & Team**: Performance characteristics
- **Tire Compounds**: Degradation rates and compound types
- **Track Conditions**: Temperature, weather, track evolution
- **Fuel Effects**: Load impact on lap times
- **Stint Analysis**: Position within stint, stint length
- **Qualifying Performance**: Pace differential
- **Position Effects**: Traffic and aerodynamic impact

## 🎛️ Configuration Options

### Strategy Parameters
- **Compounds**: SOFT, MEDIUM, HARD, INTERMEDIATE, WET
- **Stint Range**: 2-5 tire stints per race
- **Pit Windows**: Configurable early/late pit stop limits
- **Minimum Stint Length**: 3 laps minimum
- **Pit Stop Time**: Adjustable penalty (default: 22 seconds)

### Environmental Settings
- **Track Temperature**: 15-60°C
- **Air Temperature**: 10-45°C
- **Rain Conditions**: Dry/Wet strategy optimization
- **Track Evolution**: Automatic grip improvement modeling

## 🔧 Advanced Usage

### Custom Model Training
```python
from lapprediction import EnhancedModelPipeline

# Initialize pipeline
pipeline = EnhancedModelPipeline()

# Train with custom parameters
model, metrics = pipeline.train_and_save_enhanced_model(
    model_type='xgboost',
    optimize=True,
    save_path='custom_model.joblib'
)
```

### Strategy Generation
```python
from rec import find_best_strategy

# Generate optimal strategy
best_strategy, best_time, best_df, top_strategies = find_best_strategy(
    model=model,
    driver='VER',
    team='Red Bull Racing',
    race='Monaco',
    qual_time=75.0,
    start_pos=1,
    rain=False,
    total_laps=78,
    num_stints=3,
    pit_stop_time=22.0
)
```

## 📊 Performance Metrics

### Model Accuracy
- **RMSE**: ~0.6-1.5 seconds lap time prediction
- **MAE**: ~0.4-0.6 seconds average error
- **R²**: 0.95-0.99 correlation with actual lap times

### Strategy Optimization
- **Generation Speed**: 2000+ strategies in 30-60 seconds
- **Evaluation Accuracy**: Multi-threaded model prediction
- **Strategy Variety**: Balanced representation across stint counts

## 🧪 Testing

### Run All Tests
```bash
python3 utils/test.py
```

### Test Categories
- **Model Loading**: Validate trained model integrity
- **Data Consistency**: Check data quality and ranges
- **Feature Engineering**: Verify enhanced feature creation
- **Model Prediction**: Test prediction accuracy
- **Strategy Optimization**: End-to-end workflow validation

### Driver MSE Analysis
```bash
# Calculate per-driver prediction accuracy
python3 utils/test.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```bash
   # Retrain the model
   python3 lapprediction.py
   ```

2. **Data File Missing**
   ```bash
   # Check utils/f1_data.pkl exists
   # Regenerate if needed through data collection
   ```

3. **FastF1 Cache Issues**
   ```bash
   # Clear cache directory
   rm -rf /path/to/f1_cache
   ```

4. **Streamlit Port Conflicts**
   ```bash
   # Use different port
   streamlit run Strat_Prediction.py --server.port 8502
   ```

## 🔄 Updates and Maintenance

### Data Updates
- F1 data is cached automatically by FastF1
- Model retraining recommended after major regulation changes
- Strategy parameters updated based on real-world pit stop times

### Model Improvements
- Regular hyperparameter optimization
- Feature engineering enhancements
- Multi-algorithm ensemble methods

## 📋 Requirements

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for cache and models
- **CPU**: Multi-core recommended for strategy optimization

### Python Dependencies
See `requirements.txt` for complete list:
- **Core**: streamlit, pandas, numpy, scikit-learn
- **ML**: xgboost, lightgbm, optuna
- **F1 Data**: fastf1
- **Visualization**: plotly, matplotlib, seaborn

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

FastF1 is licensed under the MIT License - see the LICENSE file for details.

## 🏎️ Acknowledgments

- **FastF1**: For providing excellent F1 data access
- **Formula 1**: For the exciting sport that inspired this project
- **Streamlit**: For the amazing web app framework
- **XGBoost Team**: For the powerful ML algorithm

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite (`python3 utils/test.py`)
3. Create an issue in the repository

---

*"In F1, the perfect strategy doesn't exist, but with data science, we can get pretty close."*