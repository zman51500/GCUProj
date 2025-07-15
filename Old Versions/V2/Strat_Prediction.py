import streamlit as st
import pandas as pd
import fastf1
import os
from joblib import load
from rec import find_best_strategy  # your recommendation engine module

# Setup basic data
races = list(fastf1.get_event_schedule(2025, include_testing=False)['EventName'])
teams = ['Red Bull Racing', 'Alpine', 'Mercedes', 'Aston Martin', 'Ferrari',
         'Racing Bulls', 'Williams', 'Kick Sauber', 'Haas F1 Team', 'McLaren']

drivers = {
    'McLaren': ['NOR', 'PIA'],
    'Ferrari': ['LEC', 'HAM'],
    'Mercedes': ['RUS', 'ANT'],
    'Red Bull Racing': ['VER', 'TSU'],
    'Williams': ['ALB', 'SAI'],
    'Kick Sauber': ['HUL', 'BOR'],
    'Racing Bulls': ['LAW', 'HAD'],
    'Aston Martin': ['ALO', 'STR'],
    'Haas F1 Team': ['OCO', 'BEA'],
    'Alpine': ['GAS', 'COL']
}

compound_options = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']

model_path = os.path.join(os.path.dirname(__file__), 'utils', 'lapprediction_model.joblib')
model = load(model_path)

# Initialize session state variables
if "recommended_strategy" not in st.session_state:
    st.session_state.recommended_strategy = None
if "num_stints" not in st.session_state:
    st.session_state.num_stints = 2
if "reset_stints" not in st.session_state:
    st.session_state.reset_stints = False

# Handle reset flag before widget creation
if st.session_state.reset_stints:
    st.session_state.num_stints = 2
    st.session_state.recommended_strategy = None
    st.session_state.reset_stints = False

# Sync num_stints with recommended strategy before rendering slider
if st.session_state.recommended_strategy is not None:
    rec_len = len(st.session_state.recommended_strategy)
    if st.session_state.num_stints != rec_len:
        st.session_state.num_stints = rec_len
    slider_disabled = True
else:
    slider_disabled = False

# Streamlit app config
st.set_page_config(page_title="Strat1", layout="centered")

# Sidebar inputs
st.sidebar.title("🏁 Strat1 - Race Strategy Prediction 🏎️")
st.sidebar.header("Race Setup")

race = st.sidebar.selectbox('Select Race', options=races)
qual = st.sidebar.number_input("Qualifying Lap Time (s)", step=0.01, value=75.00)
start = st.sidebar.number_input("Starting Position", min_value=1, max_value=20, value=1)
total_laps = st.sidebar.number_input("Total Race Laps", min_value=2, max_value=100, value=57)
rain = st.sidebar.checkbox("Rain Expected", value=False, help="Check if rain is expected during the Race")

# Use session_state.num_stints for the slider's value and control disabling
num_stints = st.sidebar.slider(
    "Number of Tire Stints",
    min_value=2,
    max_value=5,
    value=st.session_state.num_stints,
    disabled=slider_disabled
)

st.sidebar.markdown("### Team and Driver Selection")
team = st.sidebar.selectbox("Select Team", options=teams)
driver = st.sidebar.selectbox("Select Driver", options=drivers[team])

# Button to run recommendation engine
if st.button("🔍 Recommend Optimal Strategy"):
    with st.spinner("Optimizing strategy... this may take a few moments ⏳"):
        best_strategy, best_time, best_df, top_strategies = find_best_strategy(
            model=model,
            driver=driver,
            team=team,
            race=race,
            qual_time=qual,
            start_pos=start,
            rain=rain,
            total_laps=total_laps,
            num_stints=num_stints
        )

    if best_df is not None:
        st.session_state.recommended_strategy = best_strategy
        st.success(f"🏆 Optimal strategy found! Total predicted time: {best_time:.2f} seconds")

        st.markdown("### 🥇 Top 3 Strategies")
        for idx, (t, strat, _) in enumerate(top_strategies, 1):
            stint_summary = ' → '.join([f"{c}({l})" for c, l in strat])
            st.markdown(f"**{idx}.** {stint_summary} — `{t:.2f}s`")
    else:
        st.error("❌ No valid strategy found. Try adjusting race inputs.")

if st.sidebar.button("🔄 Clear Recommended Strategy"):
    st.session_state.reset_stints = True  # flag to reset on next run

# Manual Tire Strategy Input — Auto-filled if recommendation exists
st.header("🛠️ Input Tire Strategy (Editable)")

strategy_df = pd.DataFrame()
prev_end = 0
lap = 1

# Automatically divide laps evenly among stints if needed
base_stint_length = total_laps // num_stints
extra_laps = total_laps % num_stints

for i in range(num_stints):
    st.subheader(f"Stint {i+1}")

    if i == 0:
        stint_start = 1
    else:
        stint_start = prev_end + 1

    if st.session_state.recommended_strategy and i < len(st.session_state.recommended_strategy):
        default_compound, default_length = st.session_state.recommended_strategy[i]
        default_end = stint_start + default_length - 1
    else:
        default_compound = None
        default_length = base_stint_length + (1 if i < extra_laps else 0)
        default_end = stint_start + default_length - 1

    # Adjust if stint end exceeds total_laps
    if default_end > total_laps:
        default_end = total_laps

    if stint_start > total_laps:
        st.warning(f"⚠️ Stint {i+1} starts after the total number of race laps. Reduce the number of stints or laps on previous stint.")
        break

    start_lap = st.number_input(f"Start Lap (Stint {i+1})", value=stint_start, key=f'start_{i}', disabled=True)
    end_lap = st.number_input(
        f"End Lap (Stint {i+1})",
        min_value=start_lap,
        max_value=total_laps,
        value=default_end,
        key=f'end_{i}'
    )

    compound = st.selectbox(
        f"Tire Compound (Stint {i+1})",
        compound_options,
        index=compound_options.index(default_compound) if default_compound else i % 2,
        key=f'compound_{i}'
    )

    for j in range(1, end_lap - start_lap + 2):
        row = {
            'Driver': driver,
            'Team': team,
            'Compound': compound,
            'FreshTyre': j == 1,
            'PitLap': j == 1 and lap != 1,
            'EventName': race,
            'EventYear': 2025,
            'TrackStatus': ['1'],
            'LapTime_Qualifying': qual,
            'TyreLife': j,
            'LapNumber': lap,
            'StartingPosition': start,
            'Rainfall': rain,
            'AirTemp': 25.0,
            'TrackTemp': 30.0,
            'Position': start
        }
        strategy_df = pd.concat([strategy_df, pd.DataFrame([row])], ignore_index=True)
        lap += 1

    prev_end = lap - 1

# Validate total laps coverage
last_lap = lap - 1

if last_lap < total_laps:
    st.warning(f"⚠️ Total laps entered ({last_lap}) are less than race laps ({total_laps}). Add more stints or laps.")
elif last_lap > total_laps:
    st.error(f"❌ Stint lap range exceeds total race laps ({total_laps}). Adjust your strategy.")
elif not rain and len(set(strategy_df['Compound'])) < 2:
    st.warning("⚠️ At least 2 different compounds are required in dry weather. Adjust your strategy ")
else:
    pred = model.predict(strategy_df)
    st.success("✅ Strategy accepted. Showing predicted lap times.")
    st.success(f"Total predicted time: {pred.sum():.2f} seconds")

    display_df = strategy_df[['LapNumber', 'Compound', 'TyreLife']].copy()
    display_df.set_index('LapNumber', inplace=True)
    display_df['Predicted Lap Time (s)'] = pred
    st.dataframe(display_df, use_container_width=True)

    show = pd.DataFrame({
        'LapNumber': strategy_df['LapNumber'],
        'Compound': strategy_df['Compound'],
        'LapTime': pred,
    })

    st.line_chart(show, x='LapNumber', y='LapTime', use_container_width=True, height=400)
