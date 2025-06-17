from strategy import simulate_strategy
from utils.fastf1_data import get_driver_pace, get_avg_pit_loss
from utils.recommender import recommend_strategy
import pandas as pd

year = 2024
gp = 'Australia'
driver = 'NOR'

#base_lap_time = get_driver_pace(year, gp, driver)
#pit_loss = get_avg_pit_loss(year-1, gp).mean().total_seconds()

strategy = recommend_strategy(total_laps=58, weather='Dry', year=year, gp=gp)
print("Recommended Strategy:")
for stint in strategy:
    print(f"Tire: {stint['tire']}, Laps: {stint['laps']}")