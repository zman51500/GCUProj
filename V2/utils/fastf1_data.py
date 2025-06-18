
import fastf1

fastf1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')

def get_driver_pace(year, gp, driver_code):
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    driver = session.laps.pick_driver(driver_code)
    return driver['LapTime'].mean().total_seconds()

def get_avg_pit_loss(year, gp):
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    pit_laps = session.laps[session.laps['PitInTime'].notnull()]
    return pit_laps['PitOutTime'] - pit_laps['PitInTime']
