import streamlit as st
import os
import pickle
import numpy as np
import fastf1 as f1
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Strat1", layout="centered")

with open(os.path.join(os.path.dirname(__file__).replace('pages',''), 'utils', 'f1_data.pkl'), 'rb') as f:
    comp = pickle.load(f)

    # Load events and filter completed races
    events = f1.get_event_schedule(2025, include_testing=False)
    completed_events = events[pd.to_datetime('today', utc=True) > pd.to_datetime(events['Session5DateUtc'], utc=True)]
    event_names = completed_events['EventName'].tolist()

    st.title("2025 F1 Race Analysis")

    st.sidebar.title("🏁 Strat1 - Race Analysis 🏎️")
    st.sidebar.header("Race Options")
    selected_race = st.sidebar.selectbox("Select a completed race:", event_names, index=0)


    @st.cache_data(show_spinner=True)
    def load_data(race_name):
        data = comp[(comp['EventName'] == race_name) & (comp['EventYear'] == 2025)]
        results = data[['ClassifiedPosition', 'Driver', 'DriverNumber', 'Points']]
        laps = data[['Driver','LapTime', 'LapNumber', 'Compound', 'Position', 'PitLap']]
        return results, laps, data['EventName'].unique()[0]
    

    results_df, laps_df, event_title = load_data(selected_race)

    fastest_lap = (laps_df.groupby('Driver')['LapTime'].min()).rename("FastestLap (s)")
    fastest_lap = fastest_lap.fillna('No Laps Completed')
    res = results_df.set_index('Driver').join(fastest_lap).drop_duplicates()
    res['ClassifiedPosition'] = pd.to_numeric(res['ClassifiedPosition'], errors = 'coerce')
    res = res.sort_values(['Points', 'ClassifiedPosition', 'FastestLap (s)'], ascending = [False, True, True])
    res['ClassifiedPosition'] = res['ClassifiedPosition'].fillna('DNF')
    st.subheader(f"Race Data for 2025 {event_title}")
    st.dataframe(res)

    st.sidebar.subheader("Driver Selection")
    selected_drivers = st.sidebar.multiselect(
        "Select Drivers",
        options=results_df['Driver'].unique(),
    )

    st.sidebar.subheader('Method Selection')
    method = st.sidebar.selectbox("Select Method:", ['Place Chart', 'Lap Time Box Plot', 'Lap Time Chart'])

      # Chart Generators
    def gen_place_chart(df):
        fig = px.scatter(df, x='LapNumber', y='Position', color='Driver',
                         title=f"{event_title} F1 Place Chart",
                         color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(paper_bgcolor='black', font_color='#7FDBFF',
                          xaxis_title='Lap Number', yaxis_title='Position',
                          yaxis=dict(autorange='reversed'))
        return fig

    def gen_lap_time_chart(df):
        fig = px.scatter(df, x='LapNumber', y='LapTime', color='Driver',
                         hover_data=['Compound', 'PitLap'],
                         title=f"{event_title} F1 Lap Time Chart")
        fig.update_traces(mode='lines+markers')
        fig.update_layout(paper_bgcolor='black', font_color='#7FDBFF',
                          xaxis_title='Lap Number', yaxis_title='Lap Time (s)')
        return fig

    def gen_box_chart(df):
        ind = np.where(df['PitLap'] == False)
        df = df.iloc[ind]
        fig = px.box(df, x='Driver', y='LapTime', color='Driver',
                     hover_data=['LapNumber', 'Compound'],
                     title=f"{event_title} F1 Lap Time Box Plot")
        fig.update_layout(paper_bgcolor='black', font_color='#7FDBFF')
        return fig
    
      # Filter and Plot
    if selected_drivers:
        filtered_laps = laps_df[laps_df['Driver'].isin(selected_drivers)]
        if method == 'Place Chart':
            st.plotly_chart(gen_place_chart(filtered_laps), use_container_width=True)
        elif method == 'Lap Time Chart':
            st.plotly_chart(gen_lap_time_chart(filtered_laps), use_container_width=True)
        elif method == 'Lap Time Box Plot':
            st.plotly_chart(gen_box_chart(filtered_laps), use_container_width=True)
    else:
        st.info("Please select at least one driver to display the chart.")


