import fastf1

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.encoder import MultiHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from joblib import dump
import pickle
fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')


# Model pipeline setup
catfeatures = ['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear','Rainfall']
status = ['TrackStatus']
standard = ['LapTime_Qualifying', 'AirTemp', 'TrackTemp']
numerical_features = ['TyreLife', 'LapNumber', 'LapTime', 'StartingPosition','Position']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), catfeatures),
        ('status', MultiHotEncoder(), status),
        ('num', StandardScaler(), standard)

    ],
    remainder='passthrough'
)

clf = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000, 
        learning_rate = .2,
        max_depth=7,
        random_state=100
        )

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', clf)
    ]
)
with open('./utils/f1_data.pkl', 'rb') as f:
    comp = pickle.load(f)
comp = comp[['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear','Rainfall',
             'TrackStatus', 'LapTime_Qualifying', 'AirTemp', 'TrackTemp', 'TyreLife', 'LapNumber',
               'LapTime', 'StartingPosition','Position']]
comp = comp.dropna()

#Train Model
X = comp.drop(columns=['LapTime'])
y = comp['LapTime']

model.fit(X, y)

dump(model, 'utils/lapprediction_model.joblib')

