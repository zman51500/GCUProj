import streamlit as st
import pandas as pd
import pickle
import fastf1
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
    'Haas': ['OCO','BEA'],
    'Alpine': ['GAS','COL']
}

with open('V2/utils/lapprediction_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_lap_times(strategy):
    """
    Mock prediction: base time + compound adjustment + stint penalty.
    Replace this with your real ML model.
    """
    lap_times = []
    for stint in strategy:
        start, end, compound = stint['start_lap'], stint['end_lap'], stint['compound']
        for lap in range(start, end + 1):
            base_time = 90  # base lap time in seconds
            compound_penalty = {'SOFT': 0, 'MEDIUM': 1.5, 'HARD': 3.0}[compound]
            tire_wear_penalty = (lap - start) * 0.2  # gets slower per lap
            lap_time = base_time + compound_penalty + tire_wear_penalty
            lap_times.append({'Lap': lap, 'Compound': compound, 'Predicted Lap Time (s)': round(lap_time, 2)})
    return lap_times

# --- Streamlit App ---
st.set_page_config(page_title="F1 Tire Strategy Predictor", layout="centered")

st.title("🏎️ Formula 1 Tire Strategy Lap Time Predictor")

st.sidebar.header("Race Setup")
race = st.sidebar.selectbox('Select Race',options=races)
qual = st.sidebar.number_input("Qualifying Lap Time (s)", min_value=60, max_value=120, value=90),
start = st.sidebar.number_input("Starting Position", min_value=1, max_value=20, value=1)
total_laps = st.sidebar.number_input("Total Race Laps", min_value=1, max_value=100, value=58)
num_stints = st.sidebar.slider("Number of Tire Stints", min_value=1, max_value=5, value=3)
st.sidebar.markdown("### Team and Driver Selection")
team = st.sidebar.selectbox("Select Team", options=teams)
driver = st.sidebar.selectbox("Select Driver", options=drivers[team])

st.header("🛠️ Input Tire Strategy")
st.markdown("Use the dropdown menus and lap ranges to define each tire stint.")

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
    start_lap = st.number_input(f"Start Lap (Stint {i+1})", min_value=1, max_value=total_laps, value=prev_end + 1, key=f'start_{i}')
    if i+1 == num_stints:
        end = total_laps
    else:
        end = min(start_lap + 25, total_laps)
    end_lap = st.number_input(f"End Lap (Stint {i+1})", min_value=start_lap + 1, max_value=total_laps,
                               value=end, key=f'end_{i}')
    compound = st.selectbox(f"Tire Compound (Stint {i+1})", compound_options, key=f'compound_{i}')

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
            'StartingPosition': int(start)
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
        'LapTime': pred,
    }, index=df['LapNumber'])

    st.line_chart(show)