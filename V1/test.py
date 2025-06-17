from joblib import load
from model import load_enhanced_race_data
import fastf1
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')

mod = load('mod.joblib')

comp = ['Australian Grand Prix']
df = pd.concat([load_enhanced_race_data(2025, g) for g in comp])

# Clean and preprocess
df = df.dropna()

# Encode categorical features
df = pd.get_dummies(df, columns=['strategy', 'team'])

# Define features and label
features = [col for col in df.columns if col not in ['driver', 'finish_pos']]
X = df[features]
y = df['finish_pos']

mod.predict(X)

fastf1.Cache.offline_mode()