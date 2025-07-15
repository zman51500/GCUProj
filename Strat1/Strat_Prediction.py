"""
F1 Strategy Prediction Streamlit App - Enhanced

This application provides an interactive interface for F1 tire strategy optimization.
Users can input race parameters and get optimal tire strategy recommendations with enhanced features.
"""

import streamlit as st
import pandas as pd
import fastf1
import os
import numpy as np
from joblib import load
from typing import Dict, List, Optional

from rec import find_best_strategy

# Configuration constants
CURRENT_YEAR = 2025
DEFAULT_TEMP = {'air': 25.0, 'track': 30.0}
COMPOUND_OPTIONS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']

# F1 2025 data
TEAMS = [
    'Red Bull Racing', 'Alpine', 'Mercedes', 'Aston Martin', 'Ferrari',
    'Racing Bulls', 'Williams', 'Kick Sauber', 'Haas F1 Team', 'McLaren'
]

DRIVERS = {
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


@st.cache_data
def load_race_calendar() -> List[str]:
    """Load and cache the F1 race calendar."""
    try:
        schedule = fastf1.get_event_schedule(CURRENT_YEAR, include_testing=False)
        return list(schedule['EventName'])
    except Exception as e:
        st.error(f"Failed to load race calendar: {e}")
        return ["Monaco", "Silverstone", "Monza"]  # Fallback races


@st.cache_resource
def load_prediction_model():
    """Load and cache the prediction model."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'utils', 'lapprediction_model.joblib')
        return load(model_path)
    except Exception as e:
        st.error(f"Failed to load prediction model: {e}")
        return None


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "recommended_strategy": None,
        "num_stints": 2,
        "reset_stints": False,
        "pit_stop_time": 22.0,
        "is_optimizing": False,
        "optimization_results": None,
        "show_results": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_strategy_reset():
    """Handle the strategy reset logic."""
    if st.session_state.reset_stints:
        st.session_state.num_stints = 2
        st.session_state.recommended_strategy = None
        st.session_state.optimization_results = None
        st.session_state.show_results = False
        st.session_state.reset_stints = False


def sync_stints_with_recommendation():
    """Sync number of stints with recommended strategy."""
    slider_disabled = False
    
    if st.session_state.recommended_strategy is not None:
        rec_len = len(st.session_state.recommended_strategy)
        if st.session_state.num_stints != rec_len:
            st.session_state.num_stints = rec_len
        slider_disabled = True
    
    return slider_disabled


def render_sidebar() -> Dict:
    """Render the sidebar with race configuration options."""
    st.sidebar.title("🏁 Strat1 - Race Strategy Prediction 🏎️")
    st.sidebar.header("Race Setup")
    
    # Load data
    races = load_race_calendar()
    
    # Race configuration
    race_config = {
        'race': st.sidebar.selectbox('Select Race', options=races),
        'qual_time': st.sidebar.number_input(
            "Qualifying Lap Time (s)", 
            step=0.01, 
            value=75.00,
            min_value=55.0,
            max_value=120.0,
            help="Best qualifying lap time for reference"
        ),
        'start_pos': st.sidebar.number_input(
            "Starting Position", 
            min_value=1, 
            max_value=20, 
            value=1
        ),
        'total_laps': st.sidebar.number_input(
            "Total Race Laps", 
            min_value=2, 
            max_value=100, 
            value=57
        )
    }
    
    # Stint configuration
    slider_disabled = sync_stints_with_recommendation()
    race_config['num_stints'] = st.sidebar.slider(
        "Number of Tire Stints",
        min_value=2,
        max_value=5,
        value=st.session_state.num_stints,
        disabled=slider_disabled,
        help="Number of tire changes + 1. If not able to adjust, reset the strategy first."
    )
    
    # Pit Stop Configuration
    st.sidebar.markdown("### Pit Stop Settings")
    race_config['pit_stop_time'] = st.sidebar.slider(
        "Pit Stop Time (seconds)",
        min_value=18.0,
        max_value=30.0,
        value=st.session_state.pit_stop_time,
        step=0.5,
        help="Time lost per pit stop including entry, stop, and exit"
    )
    
    # Update session state
    st.session_state.pit_stop_time = race_config['pit_stop_time']
    
    # Environmental conditions section
    st.sidebar.markdown("### Environmental Conditions")
    race_config['rain'] = st.sidebar.checkbox(
        "Rain Expected", 
        value=False, 
        help="Check if rain is expected during the race"
    )
    
    race_config['track_temp'] = st.sidebar.slider(
        "Track Temperature (°C)", 
        min_value=15, 
        max_value=60, 
        value=30,
        help="Track surface temperature affects tire performance"
    )
    
    race_config['air_temp'] = st.sidebar.slider(
        "Air Temperature (°C)", 
        min_value=10, 
        max_value=45, 
        value=25,
        help="Ambient air temperature"
    )
    
    # Team and driver selection
    st.sidebar.markdown("### Team and Driver Selection")
    race_config['team'] = st.sidebar.selectbox("Select Team", options=TEAMS)
    race_config['driver'] = st.sidebar.selectbox(
        "Select Driver", 
        options=DRIVERS[race_config['team']]
    )
    
    return race_config

def run_strategy_optimization(model, config: Dict):
    """Run the strategy optimization and display results."""
    # Create two columns for the buttons
    col1, col2 = st.columns([3, 1])
    
    # Check if optimization is running
    is_optimizing = st.session_state.get('is_optimizing', False)
    
    with col1:
        recommend_button = st.button(
            "🔍 Recommend Optimal Strategy" if not is_optimizing else "⏳ Optimizing...",
            disabled=is_optimizing,
            help="Generate optimal tire strategy recommendations" if not is_optimizing else "Strategy optimization in progress..."
        )
    
    with col2:
        reset_button = st.button(
            "🔄 Reset Strategy",
            disabled=is_optimizing,
            help="Reset strategy settings" if not is_optimizing else "Cannot reset while optimizing"
        )
    
    # Handle reset button
    if reset_button and not is_optimizing:
        st.session_state.reset_stints = True
        st.session_state.optimization_results = None
        st.session_state.show_results = False
        st.rerun()
    
    # Handle recommend button
    if recommend_button and not is_optimizing:
        # Set optimization state to True
        st.session_state.is_optimizing = True
        st.session_state.show_results = False
        st.rerun()
    
    # Run optimization if state indicates it should be running
    if st.session_state.get('is_optimizing', False):
        with st.spinner("Optimizing strategy... this may take a few moments ⏳"):
            try:
                best_strategy, best_time, best_df, top_strategies = find_best_strategy(
                    model=model,
                    driver=config['driver'],
                    team=config['team'],
                    race=config['race'],
                    qual_time=config['qual_time'],
                    start_pos=config['start_pos'],
                    rain=config['rain'],
                    total_laps=config['total_laps'],
                    num_stints=config['num_stints'],
                    pit_stop_time=config['pit_stop_time']
                )
                
                # Reset optimization state
                st.session_state.is_optimizing = False
                
                if best_df is not None:
                    # Store results in session state
                    st.session_state.optimization_results = {
                        'best_strategy': best_strategy,
                        'best_time': best_time,
                        'best_df': best_df,
                        'top_strategies': top_strategies,
                        'pit_stop_time': config['pit_stop_time']
                    }
                    st.session_state.recommended_strategy = best_strategy
                    st.session_state.show_results = True
                else:
                    st.session_state.optimization_results = None
                    st.session_state.show_results = False
                    
                # Rerun to update button states and show results
                st.rerun()
                    
            except Exception as e:
                # Reset optimization state on error
                st.session_state.is_optimizing = False
                st.session_state.optimization_results = None
                st.session_state.show_results = False
                st.error(f"Error during optimization: {e}")
                st.rerun()
    
    # Display results if available
    if st.session_state.get('show_results', False) and st.session_state.get('optimization_results'):
        results = st.session_state.optimization_results
        
        # Calculate pit stop info for display
        num_pit_stops = len(results['best_strategy']) - 1
        total_pit_penalty = num_pit_stops * results['pit_stop_time']
        
        st.success(f"🏆 Optimal strategy found! Total predicted time: {results['best_time']:.2f} seconds")
        st.info(f"⏱️ Includes {num_pit_stops} pit stops @ {results['pit_stop_time']}s each = {total_pit_penalty:.1f}s penalty")
        
        # Display top strategies
        st.markdown("### 🥇 Top Strategies")
        for idx, (total_time, strategy, _) in enumerate(results['top_strategies'], 1):
            stint_summary = ' → '.join([f"{compound}({length})" for compound, length in strategy])
            pit_stops = len(strategy) - 1
            st.markdown(f"**{idx}.** {stint_summary} — `{total_time:.2f}s` ({pit_stops} stops)")
    
    elif st.session_state.get('optimization_results') is None and st.session_state.get('show_results', False):
        st.error("❌ No valid strategy found. Try adjusting race inputs.")


def calculate_stint_distribution(total_laps: int, num_stints: int) -> tuple:
    """Calculate base stint distribution and extra laps."""
    base_stint_length = total_laps // num_stints
    extra_laps = total_laps % num_stints
    return base_stint_length, extra_laps


def get_stint_defaults(stint_idx: int, stint_start: int, base_stint_length: int, extra_laps: int, total_laps: int) -> tuple:
    """Get default compound and stint length for a given stint."""
    if (st.session_state.recommended_strategy and 
        stint_idx < len(st.session_state.recommended_strategy)):
        default_compound, default_length = st.session_state.recommended_strategy[stint_idx]
        default_end = stint_start + default_length - 1
    else:
        default_compound = None
        default_length = base_stint_length + (1 if stint_idx < extra_laps else 0)
        default_end = stint_start + default_length - 1
    
    # Ensure stint doesn't exceed total laps
    default_end = min(default_end, total_laps)
    
    return default_compound, default_end


def add_enhanced_features_to_dataframe(df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """Add enhanced features to match the model's expectations - Updated to match rec.py exactly."""
    df = df.copy()
    
    # Enhanced feature engineering matching the model
    compound_degradation_rates = {
        'SOFT': 0.05,
        'MEDIUM': 0.03,
        'HARD': 0.02,
        'INTERMEDIATE': 0.02,
        'WET': 0.01
    }
    
    # Tire degradation features
    df['TireDegradation'] = df.apply(
        lambda row: compound_degradation_rates.get(row['Compound'], 0.03) * row['TyreLife'], 
        axis=1
    )
    
    # Fuel load effect 
    df['FuelLoad'] = (total_laps - df['LapNumber'] + 1) * 1.8  # Assuming 1.8 kg/lap fuel consumption
    df['FuelEffect'] = df['FuelLoad'] * 0.035
    
    # Track evolution
    df['TrackEvolution'] = np.log1p(df['LapNumber']) * 0.1
    
    # Temperature features
    df['TempDiff'] = df['TrackTemp'] - df['AirTemp']
    df['TempRatio'] = df['TrackTemp'] / (df['AirTemp'] + 1)
    
    # Qualifying gap (simplified)
    min_qual_time = df['LapTime_Qualifying'].min()
    df['QualifyingGap'] = df['LapTime_Qualifying'] - min_qual_time
    
    # Stint features
    df['StintPosition'] = df['TyreLife']
    
    # Calculate stint lengths dynamically
    stint_lengths = []
    current_stint_length = 0
    
    for _, row in df.iterrows():
        current_stint_length += 1
        if row['FreshTyre'] and current_stint_length > 1:
            # New stint started, record previous stint length
            stint_lengths.extend([current_stint_length - 1] * (current_stint_length - 1))
            current_stint_length = 1
    
    # Handle the last stint
    stint_lengths.extend([current_stint_length] * current_stint_length)
    
    # Pad or trim to match dataframe length
    if len(stint_lengths) != len(df):
        stint_lengths = stint_lengths[:len(df)] + [20] * max(0, len(df) - len(stint_lengths))
    
    df['StintLength'] = stint_lengths
    df['StintProgress'] = df['StintPosition'] / df['StintLength']
    
    # Position-based features
    df['PositionGroup'] = pd.cut(df['Position'], 
                               bins=[0, 3, 6, 10, 20], 
                               labels=['Top3', 'Top6', 'Midfield', 'Back'])
    
    # Weather condition features
    df['WeatherCondition'] = df['Rainfall'].map({True: 'Wet', False: 'Dry'})
    
    # Team tier (simplified)
    df['TeamTier'] = 'Tier2'  # Default for prediction
    
    return df


def build_stint_dataframe(stint_data: Dict, lap_counter: int, total_laps: int) -> tuple:
    """Build dataframe rows for a single stint with enhanced features."""
    rows = []
    stint_length = stint_data['end_lap'] - stint_data['start_lap'] + 1
    
    for j in range(1, stint_length + 1):
        # Dynamic track temperature variation
        dynamic_track_temp = stint_data.get('track_temp', DEFAULT_TEMP['track']) + np.sin(lap_counter / 10) * 2
        
        row = {
            'Driver': stint_data['driver'],
            'Team': stint_data['team'],
            'Compound': stint_data['compound'],
            'FreshTyre': j == 1,
            'PitLap': j == 1 and lap_counter != 1,
            'EventName': stint_data['race'],
            'EventYear': CURRENT_YEAR,
            'TrackStatus': ['1'],
            'LapTime_Qualifying': stint_data['qual_time'],
            'TyreLife': j,
            'LapNumber': lap_counter,
            'StartingPosition': stint_data['start_pos'],
            'Rainfall': stint_data['rain'],
            'AirTemp': stint_data.get('air_temp', DEFAULT_TEMP['air']),
            'TrackTemp': dynamic_track_temp,
            'Position': stint_data['start_pos']
        }
        rows.append(row)
        lap_counter += 1
    
    return rows, lap_counter


def render_manual_strategy_input(config: Dict) -> Optional[pd.DataFrame]:
    """Render the manual strategy input section with enhanced features."""
    st.header("🛠️ Input Tire Strategy (Editable)")
    
    # Display current pit stop time setting
    st.info(f"⏱️ Current pit stop penalty: {config['pit_stop_time']}s per stop")
    
    strategy_df = pd.DataFrame()
    previous_stint_end = 0
    current_lap = 1
    
    # Calculate base stint distribution
    base_stint_length, extra_laps = calculate_stint_distribution(
        config['total_laps'], config['num_stints']
    )
    
    for stint_idx in range(config['num_stints']):
        st.subheader(f"Stint {stint_idx + 1}")
        
        # Calculate stint boundaries
        stint_start = 1 if stint_idx == 0 else previous_stint_end + 1
        
        # Get default values
        default_compound, default_end = get_stint_defaults(
            stint_idx, stint_start, base_stint_length, extra_laps, config['total_laps']
        )
        
        # Validation: check if stint is valid
        if stint_start > config['total_laps']:
            st.warning(f"⚠️ Stint {stint_idx + 1} starts after the total number of race laps. "
                      f"Reduce the number of stints or laps on previous stint.")
            break
        
        # Stint configuration inputs
        col1, col2 = st.columns(2)
        
        with col1:
            start_lap = st.number_input(
                f"Start Lap (Stint {stint_idx + 1})", 
                value=stint_start, 
                key=f'start_{stint_idx}', 
                disabled=True
            )
        
        with col2:
            end_lap = st.number_input(
                f"End Lap (Stint {stint_idx + 1})",
                min_value=start_lap,
                max_value=config['total_laps'],
                value=default_end,
                key=f'end_{stint_idx}'
            )
        
        # Compound selection
        compound_index = (COMPOUND_OPTIONS.index(default_compound) 
                         if default_compound else stint_idx % 2)
        compound = st.selectbox(
            f"Tire Compound (Stint {stint_idx + 1})",
            COMPOUND_OPTIONS,
            index=compound_index,
            key=f'compound_{stint_idx}'
        )
        
        # Build stint data with enhanced features
        stint_data = {
            'start_lap': start_lap,
            'end_lap': end_lap,
            'compound': compound,
            'driver': config['driver'],
            'team': config['team'],
            'race': config['race'],
            'qual_time': config['qual_time'],
            'start_pos': config['start_pos'],
            'rain': config['rain'],
            'track_temp': config.get('track_temp', DEFAULT_TEMP['track']),
            'air_temp': config.get('air_temp', DEFAULT_TEMP['air'])
        }
        
        # Build dataframe for this stint
        stint_rows, current_lap = build_stint_dataframe(stint_data, current_lap, config['total_laps'])
        stint_df = pd.DataFrame(stint_rows)
        strategy_df = pd.concat([strategy_df, stint_df], ignore_index=True)
        
        previous_stint_end = current_lap - 1
    
    # Add enhanced features to the strategy dataframe
    if not strategy_df.empty:
        strategy_df = add_enhanced_features_to_dataframe(strategy_df, config['total_laps'])
    
    return strategy_df


def validate_strategy(strategy_df: pd.DataFrame, config: Dict) -> tuple:
    """Validate the strategy and return validation results."""
    last_lap = len(strategy_df)
    
    # Check lap count
    if last_lap < config['total_laps']:
        return False, f"⚠️ Total laps entered ({last_lap}) are less than race laps ({config['total_laps']}). Add more stints or laps."
    elif last_lap > config['total_laps']:
        return False, f"❌ Stint lap range exceeds total race laps ({config['total_laps']}). Adjust your strategy."
    
    # Check compound requirements for dry races
    if not config['rain'] and len(set(strategy_df['Compound'])) < 2:
        return False, "⚠️ At least 2 different compounds are required in dry weather. Adjust your strategy."
    
    return True, "✅ Strategy validation passed."


def validate_and_predict_strategy(model, strategy_df: pd.DataFrame, config: Dict):
    """Validate the strategy and show predictions with enhanced features - Updated to match rec.py."""
    # Validate strategy
    is_valid, message = validate_strategy(strategy_df, config)
    
    if not is_valid:
        if "❌" in message:
            st.error(message)
        else:
            st.warning(message)
        return
    
    # Make predictions
    try:
        predictions = model.predict(strategy_df)
        
        # Calculate pit stop penalties using adjustable pit stop time
        num_stints = len(strategy_df['FreshTyre'].cumsum().unique())  # Count stint changes
        num_pit_stops = max(0, num_stints - 1)  # Number of pit stops
        pit_stop_time = config['pit_stop_time']  # Use adjustable pit stop time
        total_pit_time = num_pit_stops * pit_stop_time
        
        # Calculate total times to match rec.py exactly
        pure_lap_time_total = predictions.sum()
        total_time_with_pits = pure_lap_time_total + total_pit_time
        avg_lap_time = predictions.mean()
        fuel_corrected_total = (predictions + strategy_df['FuelEffect']).sum()
        
        st.success("✅ Strategy accepted. Showing predicted lap times.")
        
        # Display enhanced metrics matching rec.py
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pure Lap Time Total", f"{pure_lap_time_total:.2f}s")
        with col2:
            st.metric("Total Time (+ Pit Stops)", f"{total_time_with_pits:.2f}s")
        with col3:
            st.metric("Average Lap Time", f"{avg_lap_time:.3f}s")
        with col4:
            st.metric("Number of Pit Stops", f"{num_pit_stops}")
        
        # Show pit stop breakdown with adjustable time
        if num_pit_stops > 0:
            st.info(f"⏱️ Pit stop penalty: {num_pit_stops} × {pit_stop_time}s = {total_pit_time:.1f}s")
        
        # Display detailed results
        display_strategy_results(strategy_df, predictions)
        
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")


def display_strategy_results(strategy_df: pd.DataFrame, predictions):
    """Display the strategy results in a formatted way with enhanced information - Updated."""
    # Create display dataframe with enhanced features
    display_df = strategy_df[['LapNumber', 'Compound', 'TyreLife', 'TireDegradation', 'FuelLoad', 'FreshTyre']].copy()
    display_df.set_index('LapNumber', inplace=True)
    display_df['Predicted Lap Time (s)'] = predictions.round(3)
    display_df['Fuel Load (kg)'] = display_df['FuelLoad'].round(1)
    display_df['New Tires'] = display_df['FreshTyre'].map({True: '🔄', False: ''})
    
    # Remove internal columns and rename for display
    display_df = display_df.drop(['TireDegradation', 'FuelLoad', 'FreshTyre'], axis=1)
    
    # Show table
    st.dataframe(display_df, use_container_width=True)
    
    # Create enhanced chart data
    chart_data = pd.DataFrame({
        'LapNumber': strategy_df['LapNumber'],
        'Compound': strategy_df['Compound'],
        'LapTime': predictions,
        'TireDegradation': strategy_df['TireDegradation'],
        'FuelEffect': strategy_df['FuelEffect']
    })
    
    st.subheader("Lap Time Progression")
    st.line_chart(
        chart_data, 
        x='LapNumber', 
        y='LapTime', 
        use_container_width=True, 
        height=300
    )
    
    # Add strategy summary
    pit_laps = strategy_df[strategy_df['FreshTyre'] == True]['LapNumber'].tolist()
    if len(pit_laps) > 1:  # Exclude first lap
        pit_laps = pit_laps[1:]  # Remove first lap (race start)
        st.subheader("Strategy Summary")
        st.write(f"**Pit stops on laps:** {', '.join(map(str, pit_laps))}")
        
        # Show stint breakdown
        stint_data = []
        current_compound = None
        stint_start = 1
        
        for _, row in strategy_df.iterrows():
            if row['FreshTyre'] and current_compound is not None:
                # End of previous stint
                stint_data.append({
                    'Stint': len(stint_data) + 1,
                    'Compound': current_compound,
                    'Laps': f"{stint_start}-{row['LapNumber']-1}",
                    'Length': row['LapNumber'] - stint_start
                })
                stint_start = row['LapNumber']
            
            current_compound = row['Compound']
        
        # Add final stint
        stint_data.append({
            'Stint': len(stint_data) + 1,
            'Compound': current_compound,
            'Laps': f"{stint_start}-{strategy_df['LapNumber'].max()}",
            'Length': strategy_df['LapNumber'].max() - stint_start + 1
        })
        
        stint_df = pd.DataFrame(stint_data)
        st.dataframe(stint_df, use_container_width=True, hide_index=True)


def main():
    """Main function to run the Streamlit app."""
    # Configure page
    st.set_page_config(
        page_title="F1 Strategy Prediction",
        page_icon="🏁",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Handle strategy reset
    handle_strategy_reset()
    
    # Load model
    model = load_prediction_model()
    if model is None:
        st.stop()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Main content area
    st.title("🏁 F1 Tire Strategy Prediction")
    st.markdown("Optimize your F1 tire strategy using advanced machine learning predictions.")
    
    # Strategy optimization section (buttons are now inside this function)
    st.header("🎯 Strategy Optimization")
    run_strategy_optimization(model, config)
    
    # Manual strategy input section
    strategy_df = render_manual_strategy_input(config)
    
    # Validate and predict
    if strategy_df is not None and not strategy_df.empty:
        validate_and_predict_strategy(model, strategy_df, config)


if __name__ == "__main__":
    main()