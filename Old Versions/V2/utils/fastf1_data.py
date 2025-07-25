
import fastf1
import pandas as pd
import numpy as np
import pickle

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
    data = pd.merge_asof(race.laps.sort_values('Time'), race.weather_data, left_on='Time', right_on='Time')
    race.laps['LapTime'] = race.laps['LapTime'].dt.total_seconds()
    qualifying.laps['LapTime'] = qualifying.laps['LapTime'].dt.total_seconds()


    pitlap = race.laps['PitOutTime'].notna()
    pits = pitlap.rename('PitLap', inplace=True)

    data = race.laps[['Time','Driver','DriverNumber','Team','LapTime','Compound','FreshTyre',
                      'TyreLife', 'LapNumber','TrackStatus','Position']]
    starting_pos = race.results[['DriverNumber','GridPosition','Points','ClassifiedPosition']]
    data['StartingPosition'] = data.merge(starting_pos, on = 'DriverNumber', how='left')['GridPosition']
    data['Points'] = data.merge(starting_pos, on = 'DriverNumber', how='left')['Points']
    data['ClassifiedPosition'] = data.merge(starting_pos, on = 'DriverNumber', how='left')['ClassifiedPosition']
    data['Driver'] = data['Driver'].astype(str)
    data['Team'] = data['Team'].astype(str)
    data['Compound'] = data['Compound'].astype(str)
    data['FreshTyre'] = data['FreshTyre'].astype(bool)
    data['TrackStatus']=[list(x) for x in data['TrackStatus']]
    data = data.merge(pits, left_index=True, right_index=True)
    fastest_laps = qualifying.laps.groupby('Driver')['LapTime'].min().reset_index()
    data['EventYear'] = event_year
    data['EventName'] = event_name
    data = data.merge(fastest_laps, on='Driver', suffixes=('', '_Qualifying'))
    data = pd.merge_asof(data.sort_values('Time'), race.weather_data[['Time', 'Rainfall', 'TrackTemp', 'AirTemp']], left_on='Time', right_on='Time')
    data.drop(columns = ['Time'], inplace=True)
    data = pd.DataFrame(data)
    return(data)
    


#Import the data and format
comp = pd.DataFrame()
events = dict(zip([2023,2024,2025], [y23, y24, y25]))
for year in events:
    for e in events[year]:
        race = fastf1.get_session(year, e, 'R')
        race.load(laps=True, telemetry=False, weather=True, messages=False)

        qualifying = fastf1.get_session(year, e, 'Q')
        qualifying.load(laps=True, telemetry=False, weather=False, messages=False)

        comp = pd.concat([comp, format_data(race, qualifying, e, year)], ignore_index=True)

with open('./utils/f1_data.pkl', 'wb') as f:
    serialized = pickle.dumps(comp)
    f.write(serialized)

comp.to_csv('f1data.csv', index = False)
