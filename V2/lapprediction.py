import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.encoder import MultiHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pickle
fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')


# Model pipeline setup
catfeatures = ['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear']
status = ['TrackStatus']
timefeat = ['LapTime_Qualifying']
numerical_features = ['TyreLife', 'LapNumber', 'LapTime', 'StartingPosition']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), catfeatures),
        ('status', MultiHotEncoder(), status),
        ('num', StandardScaler(), timefeat)

    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000, 
        learning_rate = .1,
        max_depth=5,
        random_state=100
        )
    )]
)
with open('./utils/f1_data.pkl', 'rb') as f:
    comp = pickle.load(f)

#Train Model
X = comp.drop(columns=['LapTime','DriverNumber'])
y = comp['LapTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

with open('./utils/lapprediction_model.pkl', 'wb') as f:
    serialized = pickle.dumps(model)
    f.write(serialized)

