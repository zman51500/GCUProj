
def estimate_lap_time(base_time, tire_type, lap_number, weather='Dry'):
    degradation_rates = {
        'Soft': 0.15,
        'Medium': 0.10,
        'Hard': 0.07,
        'Intermediate': 0.2,
        'Wet': 0.25
    }
    weather_penalty = {'Dry': 0, 'Wet': 5, 'Mixed': 2.5}
    return base_time + degradation_rates[tire_type] * lap_number + weather_penalty[weather]

def simulate_strategy(strategy, base_lap_time, pit_loss, total_laps=58):
    total_time = 0
    lap_counter = 0
    lap_data = []
    for stint in strategy:
        tire, stint_laps = stint['tire'], stint['laps']
        for lap in range(stint_laps):
            lap_time = estimate_lap_time(base_lap_time, tire, lap)
            lap_data.append({'Lap': lap_counter + 1, 'Time': lap_time, 'Tire': tire})
            total_time += lap_time
            lap_counter += 1
        if lap_counter < total_laps:
            total_time += pit_loss
    return total_time, lap_data
