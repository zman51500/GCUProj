import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
from matplotlib import pyplot as plt  
import numpy as np
import joblib
fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')



#Get real sessions from 23 to 25
y23 = fastf1.get_event_schedule(2023, include_testing=False)['EventName'].values
y24 = fastf1.get_event_schedule(2024, include_testing=False)['EventName'].values
y25 = fastf1.get_event_schedule(2025, include_testing=False)
y25 = y25.iloc[np.where(
    pd.to_datetime('today', utc=True) > pd.to_datetime(y25['Session5DateUtc'], utc=True)
)]['EventName'].values


#data formating function
def format_data(race, qualifying,event_name, event_year):
    race.laps['LapTime'] = race.laps['LapTime'].dt.total_seconds()
    qualifying.laps['LapTime'] = qualifying.laps['LapTime'].dt.total_seconds()


    pitlap = race.laps['PitOutTime'].notna()
    pits = pitlap.rename('PitLap', inplace=True)

    data = race.laps[['Driver','DriverNumber','Team','LapTime','Compound','FreshTyre','TyreLife', 'LapNumber','TrackStatus']]
    starting_pos = race.results[['DriverNumber','GridPosition']]
    data['StartingPosition'] = data.merge(starting_pos, on = 'DriverNumber', how='left')['GridPosition']
    data['TrackStatus']=[list(x) for x in data['TrackStatus']]
    data = data.merge(pits, left_index=True, right_index=True)
    fastest_laps = qualifying.laps.groupby('Driver')['LapTime'].min().reset_index()
    data['EventYear'] = event_year
    data['EventName'] = event_name
    data = data.merge(fastest_laps, on='Driver', suffixes=('', '_Qualifying'))
    data = data.dropna()
    data = pd.DataFrame(data)
    return(data)
    


#Import the data and format
comp = pd.DataFrame()
events = dict(zip([2023,2024,2025], [y23, y24, y25]))
for year in events:
    for e in events[year]:
        race = fastf1.get_session(year, e, 'R')
        race.load(laps=True, telemetry=False, weather=False, messages=False)

        qualifying = fastf1.get_session(year, e, 'Q')
        qualifying.load(laps=True, telemetry=False, weather=False, messages=False)

        comp = pd.concat([comp, format_data(race, qualifying, e, year)], ignore_index=True)


#Custom transformer for multi-hot encoding
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """
    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()

    def fit(self, X:pd.DataFrame, y=None):
        for i in range(X.shape[1]): # X can be of multiple columns
            mlb = MultiLabelBinarizer()
            mlb.fit(X.iloc[:,i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.n_columns += 1
        return self

    def transform(self, X:pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:,i]))

        result = np.concatenate(result, axis=1)
        return result


# Model pipeline setup
catfeatures = ['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear']
status = ['TrackStatus']
timefeat = ['LapTime_Qualifying']
numerical_features = ['TyreLife', 'LapNumber', 'LapTime', 'StartingPosition', 'DriverNumber']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), catfeatures),
        ('status', MultiHotEncoder(), status),
        ('num', StandardScaler(), timefeat)

    ],
    remainder='passthrough'  # Keep numerical features as is
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

#Train Model
X = comp.drop(columns=['LapTime'])
y = comp['LapTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse:.2f} seconds')

plt.scatter(y_test.index, y_test-y_pred)
plt.show()

for d in X['Driver'].unique():
    pred = model.predict(X[X['Driver'] == d])
    se = (y[X['Driver'] == d] - pred ).mean()
    RMSE = root_mean_squared_error(pred, y[X['Driver'] == d].values)
    print(f'Driver {d} RMSE: {RMSE} || SE: {se}')

joblib.dump(model, 'lapprediction_model.pkl')
