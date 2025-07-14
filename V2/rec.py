import pandas as pd
from itertools import combinations, product
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def generate_strategies(total_laps, compounds, min_stints=2, max_stints=4, max_strategies=1500, rain=False, fixed_stints=None):
    def create_strategy(stints, stint_lengths, compounds, rain):
        strategies = []
        for comp in product(set(compounds), repeat=len(stints) - 1):
            if not rain and len(set(comp)) < 2:
                continue  # Require at least two different compounds for dry races
            strategy = list(zip(comp, stint_lengths))
            strategies.append(strategy)
        return strategies

    strategies = []
    stint_range = [fixed_stints] if fixed_stints else range(min_stints, max_stints + 1)
    for stint_count in stint_range:
        for splits in combinations(range(2, total_laps), stint_count - 1):
            stints = [0] + list(splits) + [total_laps]
            stint_lengths = [stints[i+1] - stints[i] for i in range(len(stints)-1)]
            strategies.extend(create_strategy(stints, stint_lengths, compounds, rain))
            if len(strategies) >= max_strategies:
                return strategies[:max_strategies]
    return strategies


def build_strategy_df(strategy, driver, team, race, qual_time, start_pos, rain, year=2025):
    df = pd.DataFrame()
    lap = 1
    for compound, duration in strategy:
        for i in range(1, duration + 1):
            row = {
                'Driver': driver,
                'Team': team,
                'Compound': compound,
                'FreshTyre': i == 1,
                'PitLap': i == 1 and lap != 1,
                'EventName': race,
                'EventYear': year,
                'TrackStatus': ['1'],
                'LapTime_Qualifying': qual_time,
                'TyreLife': i,
                'LapNumber': lap,
                'StartingPosition': start_pos,
                'Rainfall': rain,
                'AirTemp': 25.0,
                'TrackTemp': 30.0,
                'Position': start_pos
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            lap += 1
    return df


def evaluate_strategy(model, strategy, driver, team, race, qual_time, start_pos, rain):
    try:
        df = build_strategy_df(strategy, driver, team, race, qual_time, start_pos, rain)
        pred = model.predict(df)
        total_time = pred.sum()
        df['Predicted Lap Time'] = pred
        return total_time, strategy, df
    except Exception:
        return np.inf, strategy, None


def find_best_strategy(model, driver, team, race, qual_time, start_pos, rain, total_laps, num_stints=None):
    if rain:
        compounds = ['INTERMEDIATE', 'WET']
    else:
        compounds = ['SOFT', 'MEDIUM', 'HARD']

    strategies = generate_strategies(total_laps, compounds, rain=rain, fixed_stints=num_stints)

    results = []
    for result in tqdm(Parallel(n_jobs=-1)(
        delayed(evaluate_strategy)(model, strategy, driver, team, race, qual_time, start_pos, rain)
        for strategy in strategies
    ), total=len(strategies), desc="Evaluating Strategies"):
        results.append(result)

    valid_results = [r for r in results if r[2] is not None]
    if not valid_results:
        return None, np.inf, None, []

    valid_results.sort(key=lambda x: x[0])  # Sort by total time
    best_time, best_strategy, best_df = valid_results[0]
    top_strategies = []
    seen_strategies = []
    for total_time, strategy, df in valid_results:
        simplified = [(c, l) for c, l in strategy]
        if not any(sum(abs(a[1] - b[1]) for a, b in zip(simplified, seen)) < 20 and [a[0] for a in simplified] == [b[0] for b in seen] for seen in seen_strategies):
            top_strategies.append((total_time, strategy, df))
            seen_strategies.append(simplified)
        if len(top_strategies) >= 3:
            break  # Top 3

    return best_strategy, best_time, best_df, top_strategies
