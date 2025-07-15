import streamlit as st
import os
import pickle
import numpy as np
import fastf1 as f1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure page to match Strategy Prediction style
st.set_page_config(
    page_title="F1 Race Analysis",
    page_icon="🏁",
    layout="wide"
)

# Load data
@st.cache_data
def load_f1_data():
    """Load and cache F1 data."""
    try:
        with open(os.path.join(os.path.dirname(__file__).replace('pages',''), 'utils', 'f1_data.pkl'), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load F1 data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_completed_races():
    """Load and cache completed races list."""
    try:
        events = f1.get_event_schedule(2025, include_testing=False)
        completed_events = events[pd.to_datetime('today', utc=True) > pd.to_datetime(events['Session5DateUtc'], utc=True)]
        return completed_events['EventName'].tolist()
    except Exception as e:
        st.error(f"Failed to load race calendar: {e}")
        return ["Monaco", "Silverstone", "Monza"]  # Fallback

def render_sidebar():
    """Render the sidebar with race analysis options."""
    st.sidebar.title("🏁 Strat1 - Race Analysis 🏎️")
    st.sidebar.header("Analysis Options")
    
    # Load data
    event_names = load_completed_races()
    
    # Race selection
    selected_race = st.sidebar.selectbox(
        "Select a completed race:", 
        event_names, 
        index=0,
        help="Choose a race from the 2025 F1 season"
    )
    
    # Analysis type
    st.sidebar.markdown("### Analysis Type")
    analysis_method = st.sidebar.selectbox(
        "Select Analysis Method:", 
        ['Position Progression', 'Lap Time Analysis', 'Performance Distribution'],
        help="Choose the type of analysis to display"
    )
    
    return selected_race, analysis_method

@st.cache_data(show_spinner=True)
def load_race_data(race_name):
    """Load and process data for a specific race."""
    comp = load_f1_data()
    
    if comp.empty:
        return pd.DataFrame(), pd.DataFrame(), "No Data"
    
    data = comp[(comp['EventName'] == race_name) & (comp['EventYear'] == 2025)]
    
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), race_name
    
    # Process results data
    results = data[['ClassifiedPosition', 'Driver', 'DriverNumber', 'Points', 'Team']].drop_duplicates()
    
    # Process lap data
    laps = data[['Driver', 'Team', 'LapTime', 'LapNumber', 'Compound', 'Position', 'PitLap', 'TyreLife']]
    
    return results, laps, data['EventName'].iloc[0]

def create_enhanced_results_table(results_df, laps_df):
    """Create an enhanced results table with additional statistics, indexed by classified position."""
    if results_df.empty or laps_df.empty:
        return pd.DataFrame()
    
    # Calculate fastest lap per driver
    fastest_lap = laps_df.groupby('Driver')['LapTime'].min().rename("Fastest Lap (s)")
    
    # Calculate average lap time (excluding pit laps)
    avg_lap = (laps_df[laps_df['PitLap'] == False]
               .groupby('Driver')['LapTime'].mean()
               .rename("Avg Lap Time (s)"))
    
    
    # Count pit stops properly - count tire changes, not pit lane entries
    def count_pit_stops_for_driver(driver_data):
        """Count actual pit stops by looking for tire changes after lap 1."""
        pit_laps = driver_data[driver_data['PitLap'] == True]['LapNumber'].tolist()
        # Remove lap 1 if it exists (race start)
        actual_pit_stops = [lap for lap in pit_laps if lap > 1]
        return len(actual_pit_stops)
    
    # Apply the counting function to each driver
    pit_stops_data = []
    for driver in laps_df['Driver'].unique():
        driver_data = laps_df[laps_df['Driver'] == driver]
        pit_count = count_pit_stops_for_driver(driver_data)
        pit_stops_data.append({'Driver': driver, 'Pit Stops': pit_count})
    
    pit_stops_df = pd.DataFrame(pit_stops_data).set_index('Driver')['Pit Stops']
    
    # DEBUG: Print pit stop counts for verification
    print("DEBUG: Calculated pit stops per driver:")
    for driver, count in pit_stops_df.items():
        print(f"  {driver}: {count} pit stops")
    
    # Combine all data
    enhanced_results = results_df.set_index('Driver').join([fastest_lap, avg_lap, pit_stops_df], how='left')
    enhanced_results = enhanced_results.drop_duplicates()
    
    # Clean and sort data
    enhanced_results['ClassifiedPosition'] = pd.to_numeric(enhanced_results['ClassifiedPosition'], errors='coerce')
    enhanced_results = enhanced_results.sort_values(
        ['Points', 'ClassifiedPosition', 'Fastest Lap (s)'], 
        ascending=[False, True, True]
    )
    enhanced_results['ClassifiedPosition'] = enhanced_results['ClassifiedPosition'].fillna('DNF')
    enhanced_results['Pit Stops'] = enhanced_results['Pit Stops'].fillna(0).astype(int)
    
    # Round numerical columns
    for col in ['Fastest Lap (s)', 'Avg Lap Time (s)']:
        if col in enhanced_results.columns:
            enhanced_results[col] = enhanced_results[col].round(3)
    
    # Reset index to get Driver as a column again, then set ClassifiedPosition as index
    enhanced_results = enhanced_results.reset_index()
    enhanced_results = enhanced_results.set_index('ClassifiedPosition')
    
    return enhanced_results

def create_position_chart(laps_df, selected_drivers, event_title):
    """Create an enhanced position progression chart."""
    if laps_df.empty:
        return go.Figure()
    
    filtered_data = laps_df[laps_df['Driver'].isin(selected_drivers)] if selected_drivers else laps_df
    
    fig = px.line(
        filtered_data, 
        x='LapNumber', 
        y='Position', 
        color='Driver',
        title=f"🏎️ {event_title} - Position Progression",
        markers=True,
        hover_data=['Compound', 'PitLap']
    )
    
    # Enhanced styling to match Strategy Prediction
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(font=dict(size=18, color='white')),
        xaxis=dict(
            title='Lap Number',
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title='Position',
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True,
            autorange='reversed'  # Lower positions at top
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        height=500
    )
    
    # Add pit stop markers
    pit_laps = filtered_data[filtered_data['PitLap'] == True]
    if not pit_laps.empty:
        for _, pit in pit_laps.iterrows():
            fig.add_annotation(
                x=pit['LapNumber'],
                y=pit['Position'],
                text="🔄",
                showarrow=False,
                font=dict(size=12)
            )
    
    return fig

def create_laptime_chart(laps_df, selected_drivers, event_title):
    """Create an enhanced lap time progression chart."""
    if laps_df.empty:
        return go.Figure()
    
    filtered_data = laps_df[laps_df['Driver'].isin(selected_drivers)] if selected_drivers else laps_df
    
    fig = px.scatter(
        filtered_data, 
        x='LapNumber', 
        y='LapTime', 
        color='Driver',
        symbol='Compound',
        title=f"⏱️ {event_title} - Lap Time Progression",
        hover_data=['Compound', 'PitLap', 'TyreLife', 'Position']
    )
    
    # Add trend lines
    fig.update_traces(mode='markers+lines')
    
    # Enhanced styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(font=dict(size=18, color='white')),
        xaxis=dict(
            title='Lap Number',
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title='Lap Time (seconds)',
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        height=500
    )
    
    return fig

def create_performance_distribution(laps_df, selected_drivers, event_title):
    """Create performance distribution charts."""
    if laps_df.empty:
        return go.Figure()
    
    filtered_data = laps_df[laps_df['Driver'].isin(selected_drivers)] if selected_drivers else laps_df
    
    # Remove pit laps for cleaner distribution
    clean_data = filtered_data[filtered_data['PitLap'] == False]
    
    fig = px.box(
        clean_data, 
        x='Driver', 
        y='LapTime', 
        color='Driver',
        title=f"📊 {event_title} - Lap Time Distribution",
        hover_data=['LapNumber', 'Compound', 'Position']
    )
    
    # Enhanced styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(font=dict(size=18, color='white')),
        xaxis=dict(
            title='Driver',
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title='Lap Time (seconds)',
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        height=500
    )
    
    return fig

def display_race_statistics(results_df, laps_df):
    """Display key race statistics in metrics."""
    if results_df.empty or laps_df.empty:
        return
    
    # Calculate key statistics
    total_laps = laps_df['LapNumber'].max() if not laps_df.empty else 0
    fastest_overall = laps_df['LapTime'].min() if not laps_df.empty else 0
    fastest_driver = laps_df.loc[laps_df['LapTime'].idxmin(), 'Driver'] if not laps_df.empty else "N/A"
    total_drivers = len(results_df['Driver'].unique()) if not results_df.empty else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Laps", total_laps)
    with col2:
        st.metric("Fastest Lap", f"{fastest_overall:.3f}s" if fastest_overall else "N/A")
    with col3:
        st.metric("Fastest Driver", fastest_driver)
    with col4:
        st.metric("Drivers", total_drivers)

def main():
    """Main function to run the Race Analysis app."""
    
    # Main content area
    st.title("🏁 2025 F1 Race Analysis")
    st.markdown("Analyze completed F1 races with detailed performance insights and visualizations.")
    
    # Render sidebar and get configuration
    selected_race, analysis_method = render_sidebar()
    
    # Load race data
    with st.spinner("Loading race data..."):
        results_df, laps_df, event_title = load_race_data(selected_race)
    
    if results_df.empty:
        st.error(f"❌ No data available for {selected_race}. Please select a different race.")
        return
    
    # Display race header
    st.header(f"🏆 {event_title} Analysis")
    
    # Display key statistics
    display_race_statistics(results_df, laps_df)
    
    # Enhanced results table
    st.subheader("📋 Race Results & Statistics")
    enhanced_results = create_enhanced_results_table(results_df, laps_df)
    if not enhanced_results.empty:
        st.dataframe(enhanced_results, use_container_width=True)
    
    # Driver selection for charts
    available_drivers = results_df['Driver'].unique().tolist()
    
    # Create columns for driver selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performance Analysis")
    
    with col2:
        selected_drivers = st.multiselect(
            "Select Drivers for Analysis:",
            options=available_drivers,
            default=available_drivers[:5] if len(available_drivers) > 5 else available_drivers,
            help="Choose drivers to include in the analysis"
        )
    
    # Display appropriate chart based on selection
    if selected_drivers:
        if analysis_method == 'Position Progression':
            fig = create_position_chart(laps_df, selected_drivers, event_title)
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_method == 'Lap Time Analysis':
            fig = create_laptime_chart(laps_df, selected_drivers, event_title)
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_method == 'Performance Distribution':
            fig = create_performance_distribution(laps_df, selected_drivers, event_title)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Please select at least one driver to display the analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Strat1 F1 Race Analysis - Powered by FastF1 and advanced data analytics*")

if __name__ == "__main__":
    main()