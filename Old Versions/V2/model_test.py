import pickle
from sklearn.metrics import root_mean_squared_error
from utils.encoder import MultiHotEncoder
from joblib import load



model = load('utils/lapprediction_model.joblib')

with open('utils/f1_data.pkl', 'rb') as f:
    comp = pickle.load(f)

comp = comp[['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear','Rainfall',
             'TrackStatus', 'LapTime_Qualifying', 'AirTemp', 'TrackTemp', 'TyreLife', 'LapNumber',
               'LapTime', 'StartingPosition','Position']]
comp = comp.dropna()

#Evaluate Model
X = comp.drop(columns=['LapTime'])
y = comp['LapTime']
pred = model.predict(X)
rmse = root_mean_squared_error(y, pred)
print(f'Root Mean Squared Error: {rmse:.2f} seconds')

for d in X['Driver'].unique():
    pred = model.predict(X[X['Driver'] == d])
    se = (y[X['Driver'] == d] - pred ).mean()
    RMSE = root_mean_squared_error(pred, y[X['Driver'] == d].values)
    print(f'Driver {d} RMSE: {RMSE:.3f} || SE: {se:.3f}')
