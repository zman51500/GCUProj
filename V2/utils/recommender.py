
import fastf1
import pandas as pd

fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')

def recommend_strategy(total_laps, weather='Dry', year=2024, gp='Monaco'):
    # Load the race session
    session = fastf1.get_session(year, gp, 'R')
    session.load()

    # Extract stint data
    laps = session.laps
    stints = laps[['Driver', 'Stint', 'Compound', 'LapNumber']].copy()
    stints = stints.groupby(['Driver', 'Stint', 'Compound']).agg({'LapNumber': ['min', 'max']})
    stints.columns = ['StartLap', 'EndLap']
    stints = stints.reset_index()
    stints['StintLength'] = stints['EndLap'] - stints['StartLap'] + 1

    # Get top drivers (finishers)
    classification = session.results
    top_finishers = classification[classification['Position'] <= 5]['Abbreviation'].values

    # Filter for top strategies
    top_stints = stints[stints['Driver'].isin(top_finishers)]
    strategy_counts = top_stints.groupby(['Driver', 'Stint']).agg({
        'Compound': 'first',
        'StintLength': 'sum'
    }).reset_index()

    # Group stints into driver-level strategies
    strategies = []
    for driver in top_stints['Driver'].unique():
        driver_stints = strategy_counts[strategy_counts['Driver'] == driver]
        strat = []
        for _, row in driver_stints.iterrows():
            strat.append({
                'tire': row['Compound'],
                'laps': int(row['StintLength'])
            })
        total_stint_laps = sum(s['laps'] for s in strat)
        if total_stint_laps < total_laps:
            strat[-1]['laps'] += total_laps - total_stint_laps
        strategies.append(strat)

    # Pick the most common strategy among top finishers
    if strategies:
        return strategies[0]
    else:
        # Fallback to heuristic
        return [
            {"tire": "Soft", "laps": total_laps // 3},
            {"tire": "Medium", "laps": total_laps // 3},
            {"tire": "Hard", "laps": total_laps - 2 * (total_laps // 3)}
        ]

t = recommend_strategy(95, weather='Dry', year=2024, gp='Monaco')
print(t)