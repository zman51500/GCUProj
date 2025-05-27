import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('./Final/f1_cache')

def load_enhanced_race_data(year, gp):
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    quali = fastf1.get_session(year, gp, 'Q')
    quali.load()

    laps = session.laps
    weather = session.weather_data
    stint_data = session.laps[['DriverNumber' ,"Stint", "Compound", "LapNumber"]]
    stint_data = stint_data.groupby(["DriverNumber", "Stint", "Compound"])
    stint_data = stint_data.count().reset_index()
    stint_data = stint_data.rename(columns={"LapNumber": "StintLength"})


    drivers = session.drivers
    data = []

    for drv in drivers:
        drv_laps = laps.pick_drivers(drv)
        drv_quali = quali.laps.pick_drivers(drv)
        drv_stints = stint_data[stint_data['DriverNumber'] == drv]

        if drv_laps.empty or drv_stints.empty or drv_quali.empty:
            continue

        # Performance metrics
        start_pos = drv_laps['Position'].values[0]
        finish_pos = drv_laps['Position'].values[-1]
        avg_lap = drv_laps['LapTime'].dropna().mean().total_seconds()

        quali_best = drv_quali['LapTime'].dropna().min().total_seconds()
        pole_time = quali.laps['LapTime'].dropna().min().total_seconds()
        quali_delta = quali_best - pole_time if quali_best else None

        strategy = drv_stints['Compound'].tolist()
        n_stops = len(strategy) - 1

        # Weather snapshot near race start
        weather_start = weather.iloc[0]
        temp_air = weather_start['AirTemp']
        temp_track = weather_start['TrackTemp']
        rain = weather_start['Rainfall']

        data.append({
            'driver': drv,
            'team': drv_laps['Team'].iloc[0],
            'start_pos': start_pos,
            'finish_pos': finish_pos,
            'avg_lap_time': avg_lap,
            'quali_delta': quali_delta,
            'num_stops': n_stops,
            'strategy': '-'.join(strategy),
            'air_temp': temp_air,
            'track_temp': temp_track,
            'rainfall': rain,
        })
    return pd.DataFrame(data)

# Load data
events = fastf1.get_event_schedule(2025, include_testing=False)
# Filter events to only include those that have occurred
comp = events.iloc[np.where(
    pd.to_datetime('today', utc=True) > pd.to_datetime(events['Session5DateUtc'], utc=True))]['EventName']
df = pd.concat([load_enhanced_race_data(2025, g) for g in comp])

# Clean and preprocess
df = df.dropna()

# Encode categorical features
df = pd.get_dummies(df, columns=['strategy', 'team'])

# Define features and label
features = [col for col in df.columns if col not in ['driver', 'finish_pos']]
X = df[features]
y = df['finish_pos']

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error in finish position: {mae:.2f}")
