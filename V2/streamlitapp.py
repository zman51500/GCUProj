import streamlit as st
import pandas as pd
import pickle
import fastf1
import os
from joblib import load
races = list(fastf1.get_event_schedule(2025,include_testing=False)['EventName'])
teams = ['Red Bull Racing', 'Alpine', 'Mercedes', 'Aston Martin', 'Ferrari',
        'Racing Bulls', 'Williams', 'Kick Sauber', 'Haas F1 Team',
        'McLaren']

drivers = {
    'McLaren': ['NOR','PIA'],
    'Ferrari': ['LEC','HAM'],
    'Mercedes': ['RUS','ANT'],
    'Red Bull Racing': ['VER','TSU'],
    'Williams': ['ALB','SAI'],
    'Kick Sauber': ['HUL', 'BOR'],
    'Racing Bulls': ['LAW','HAD'],
    'Aston Martin': ['ALO','STR'],
    'Haas F1 Team': ['OCO','BEA'],
    'Alpine': ['GAS','COL']
}

model_path = os.path.join(os.path.dirname(__file__), 'utils', 'lapprediction_model.joblib')
model = load(model_path)

# --- Streamlit App ---
st.set_page_config(page_title="Strat1", layout="centered")


st.sidebar.title("🏁 Strat1 - Race Strategy Prediction 🏎️")
st.sidebar.header("Race Setup")
race = st.sidebar.selectbox('Select Race',options=races)
qual = st.sidebar.number_input("Qualifying Lap Time (s)",step=.01, value = 90.00),
start = st.sidebar.number_input("Starting Position", min_value=1, max_value=20, value=1)
total_laps = st.sidebar.number_input("Total Race Laps", min_value=1, max_value=100, value=58)
rain = st.sidebar.checkbox("Rain Expected", value=False, help="Check if rain is expected during the Race")
num_stints = st.sidebar.slider("Number of Tire Stints", min_value=2, max_value=5, value=2)
st.sidebar.markdown("### Team and Driver Selection")
team = st.sidebar.selectbox("Select Team", options=teams)
driver = st.sidebar.selectbox("Select Driver", options=drivers[team])

st.header("🛠️ Input Tire Strategy", help = "Use the dropdown menus and lap ranges to define each tire stint.")

strategy = []
prev_end = 0
compound_options = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']

df = pd.DataFrame(columns=['Driver', 'Team', 'Compound', 'FreshTyre', 'PitLap', 'EventName', 'EventYear',
                           'TrackStatus',
                           'LapTime_Qualifying',
                           'TyreLife', 'LapNumber', 'StartingPosition'])
lap = 1
for i in range(num_stints):
    st.subheader(f"Stint {i+1}")
    if i == 0:
        start = 1
    else:
        start = prev_end + 1
    start_lap = st.number_input(f"Start Lap (Stint {i+1})",value = start, key=f'start_{i}', disabled=True)
    if i+1 == num_stints:
        end = total_laps
    else:
        end = min(start_lap + 25, total_laps)
    end_lap = st.number_input(f"End Lap (Stint {i+1})", min_value=start_lap, max_value=total_laps,
                               value=end, key=f'end_{i}')
    compound = st.selectbox(f"Tire Compound (Stint {i+1})", compound_options ,key=f'compound_{i}',
                            placeholder='Select Compound',
                            help='Select the tire compound for this stint. Options: SOFT, MEDIUM, HARD, INTERMEDIATE, WET')

    for i in range(1, end_lap - start_lap + 2):
        new = {
            'Driver': str(driver),
            'Team': str(team),
            'Compound': str(compound),
            'FreshTyre': True if i == 1 else False,
            'PitLap': True if i == 1 else False,
            'EventName': str(race),
            'EventYear': int(2025),
            'TrackStatus': ['1'],
            'LapTime_Qualifying': float(qual[0]),
            'TyreLife': int(i),
            'LapNumber': int(lap),
            'StartingPosition': int(start),
            'Rainfall': rain,
            'AirTemp': 25.0,  # Placeholder value
            'TrackTemp': 30.0  # Placeholder value
        }

        df = pd.concat([df,pd.DataFrame([new])], ignore_index=True)
        lap = lap + 1

    
    prev_end = lap - 1

# Validate total laps covered
last_lap = lap -1

if last_lap < total_laps:
    st.warning(f"⚠️ Total laps entered ({last_lap}) are less than race laps ({total_laps}). Add more stints.")
elif last_lap > total_laps:
    st.error(f"🚫 Stint lap range exceeds total race laps ({total_laps}). Adjust your strategy.")
else:
    # Predict Lap Times
    pred = model.predict(df)
    
    st.success("✅ Strategy accepted. Showing predicted lap times.")

    frame = df[['LapNumber','Compound', 'TyreLife']]
    frame.set_index('LapNumber', inplace=True)
    frame['Predicted Lap Time (s)'] = pred
    st.dataframe(frame, use_container_width=True)
    st.success(f'Total Time: {pred.sum():.2f} seconds')
    
    show = pd.DataFrame({
        'LapNumber': df['LapNumber'],
        'Compound': df['Compound'],
        'LapTime': pred,
    })

    st.line_chart(show, use_container_width=True, height=400,
                   x = 'LapNumber',y='LapTime')