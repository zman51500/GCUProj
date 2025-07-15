"""
F1 Tire Strategy Recommendation Engine V3 - Simplified

This module provides simplified functions to generate, evaluate, and optimize tire strategies
for F1 races using a trained lap time prediction model with focus on prediction accuracy.
"""

import pandas as pd
import numpy as np
from itertools import combinations, product, permutations
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any, Set
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    """Configuration class for strategy generation parameters."""
    total_laps: int
    compounds: List[str]
    min_stints: int = 2
    max_stints: int = 4 
    max_strategies: int = 2000  
    rain: bool = False
    fixed_stints: Optional[int] = None
    pit_window_start: int = 3  # Earliest pit stop lap
    pit_window_end: Optional[int] = None  # Latest pit stop lap (default: total_laps - 3)
    min_stint_length: int = 3  # Minimum laps per stint
    max_stint_length: Optional[int] = None  # Maximum laps per stint
    pit_stop_time: float = 22.0  # Adjustable pit stop penalty in seconds


@dataclass
class StrategyResult:
    """Simplified result class focused on prediction model output."""
    strategy: List[Tuple[str, int]]
    total_time: float
    average_lap_time: float
    pit_stop_time: float
    dataframe: Optional[pd.DataFrame] = None


class EnhancedFeatureBuilder:
    """Class to build enhanced features for strategy evaluation."""
    
    def __init__(self):
        self.compound_degradation_rates = {
            #Base degradation rates for each compound
            'SOFT': 0.05, # Higher degradation for soft tires
            'MEDIUM': 0.03,
            'HARD': 0.02,
            'INTERMEDIATE': 0.02,
            'WET': 0.01 # Minimal degradation for wet tires
        }
    
    def add_enhanced_features(self, df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
        """Add all enhanced features to the dataframe."""
        df = df.copy()
        
        # Tire degradation features
        df['TireDegradation'] = df.apply(
            lambda row: self._calculate_tire_degradation(row['Compound'], row['TyreLife']), 
            axis=1
        )
        
        # Fuel load effect
        df['FuelLoad'] = (total_laps - df['LapNumber'] + 1) * 1.8 # Assuming 1.8 kg/lap fuel consumption
        df['FuelEffect'] = df['FuelLoad'] * 0.035
        
        # Track evolution
        df['TrackEvolution'] = np.log1p(df['LapNumber']) * 0.1
        
        # Temperature features
        df['TempDiff'] = df['TrackTemp'] - df['AirTemp']
        df['TempRatio'] = df['TrackTemp'] / (df['AirTemp'] + 1)
        
        # Qualifying gap (use minimum qualifying time in session)
        min_qual_time = df['LapTime_Qualifying'].min()
        df['QualifyingGap'] = df['LapTime_Qualifying'] - min_qual_time
        
        # Stint features
        df = self._add_stint_features(df)
        
        # Position-based features
        df['PositionGroup'] = pd.cut(df['Position'], 
                                   bins=[0, 3, 6, 10, 20], 
                                   labels=['Top3', 'Top6', 'Midfield', 'Back'])
        
        # Weather condition features
        df['WeatherCondition'] = df['Rainfall'].map({True: 'Wet', False: 'Dry'})
        
        # Team tier (simplified for strategy evaluation)
        df['TeamTier'] = 'Tier2'  # Default tier for evaluation
        
        return df
    
    def _calculate_tire_degradation(self, compound: str, tire_life: int) -> float:
        """Calculate tire degradation effect."""
        base_rate = self.compound_degradation_rates.get(compound, 0.03)
        return base_rate * tire_life
    
    def _add_stint_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stint-related features (simplified for strategy evaluation)."""
        # For strategy evaluation, we'll use simplified stint features
        df['StintPosition'] = df['TyreLife']
        
        # Estimate stint length based on fresh tire occurrences
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
        
        return df


class SimplifiedStrategyGenerator:
    """Simplified strategy generator focused on core strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        
    def generate_strategies(self) -> List[List[Tuple[str, int]]]:
        """
        Generate strategies with focus on core patterns.
        
        Returns:
            List of strategies
        """
        strategies = []
        
        # Generate base strategies for 2, 3, and 4 stints only
        base_strategies = self._generate_base_strategies()
        strategies.extend(base_strategies)
        
        # Generate balanced strategies
        balanced_strategies = self._generate_balanced_strategies()
        strategies.extend(balanced_strategies)
        
        # Generate conservative strategies
        conservative_strategies = self._generate_conservative_strategies()
        strategies.extend(conservative_strategies)
        
        # Remove duplicates and filter invalid strategies
        strategies = self._filter_and_deduplicate(strategies)
        
        print(f"Generated {len(strategies)} unique strategies (pit stop time: {self.config.pit_stop_time}s)")
        
        return strategies[:self.config.max_strategies]
    
    def _generate_base_strategies(self) -> List[List[Tuple[str, int]]]:
        """Generate fundamental strategy patterns for 2-4 stints."""
        strategies = []
        stint_range = [self.config.fixed_stints] if self.config.fixed_stints else range(self.config.min_stints, self.config.max_stints + 1)
        
        for stint_count in stint_range:
            print(f"Generating base strategies for {stint_count} stints...")
            
            # Even distribution
            base_length = self.config.total_laps // stint_count
            extra_laps = self.config.total_laps % stint_count
            
            # Create distribution patterns
            stint_lengths = []
            for i in range(stint_count):
                length = base_length + (1 if i < extra_laps else 0)
                stint_lengths.append(length)
            
            # Generate compound combinations for this stint structure
            if self.config.rain:
                # Rain strategies - can use same compound
                compounds_to_use = self.config.compounds
                for compound_combo in product(compounds_to_use, repeat=stint_count):
                    strategy = list(zip(compound_combo, stint_lengths))
                    if self._is_valid_strategy(strategy):
                        strategies.append(strategy)
            else:
                # Dry strategies - need at least 2 different compounds
                compounds_to_use = self.config.compounds
                for compound_combo in product(compounds_to_use, repeat=stint_count):
                    if len(set(compound_combo)) >= 2:  # At least 2 different compounds
                        strategy = list(zip(compound_combo, stint_lengths))
                        if self._is_valid_strategy(strategy):
                            strategies.append(strategy)
        
        return strategies
    
    def _generate_balanced_strategies(self) -> List[List[Tuple[str, int]]]:
        """Generate strategies that balance compound types."""
        strategies = []
        
        if self.config.rain or len(self.config.compounds) < 3:
            return strategies
        
        # Three-stint strategies with one of each compound
        strategies.extend(self._generate_balanced_three_stint())
        
        # Four-stint strategies with balanced compounds
        strategies.extend(self._generate_balanced_four_stint())
        
        return strategies
    
    def _generate_balanced_three_stint(self) -> List[List[Tuple[str, int]]]:
        """Generate balanced 3-stint strategies."""
        strategies = []
        
        # Try different pit windows
        for first_pit in range(self.config.pit_window_start, min(25, self.config.total_laps // 3 + 5)):
            for second_pit in range(first_pit + self.config.min_stint_length, 
                                  min(self.config.total_laps - self.config.min_stint_length, 
                                      int(self.config.total_laps * 0.8))):
                
                first_stint = first_pit
                second_stint = second_pit - first_pit
                third_stint = self.config.total_laps - second_pit
                
                if (first_stint >= self.config.min_stint_length and 
                    second_stint >= self.config.min_stint_length and 
                    third_stint >= self.config.min_stint_length):
                    
                    stint_lengths = [first_stint, second_stint, third_stint]
                    
                    # All permutations of the three compounds
                    for compound_combo in permutations(['SOFT', 'MEDIUM', 'HARD']):
                        strategy = list(zip(compound_combo, stint_lengths))
                        if self._is_valid_strategy(strategy):
                            strategies.append(strategy)
        
        return strategies
    
    def _generate_balanced_four_stint(self) -> List[List[Tuple[str, int]]]:
        """Generate balanced 4-stint strategies."""
        strategies = []
        
        # For 4 stints, we can repeat one compound
        base_stint = self.config.total_laps // 4
        
        # Try a few variations of stint lengths
        for variation in [-1, 0, 1]:
            stint_lengths = []
            remaining_laps = self.config.total_laps
            
            for i in range(4):
                if i < 3:
                    length = max(self.config.min_stint_length, base_stint + variation)
                    length = min(length, remaining_laps - (3 - i) * self.config.min_stint_length)
                else:
                    length = remaining_laps
                
                stint_lengths.append(length)
                remaining_laps -= length
            
            if all(length >= self.config.min_stint_length for length in stint_lengths):
                # Balanced patterns: each compound used at least once, one repeated
                balanced_patterns = [
                    ['SOFT', 'MEDIUM', 'HARD', 'MEDIUM'],
                    ['MEDIUM', 'SOFT', 'HARD', 'SOFT'],
                    ['HARD', 'MEDIUM', 'SOFT', 'MEDIUM'],
                    ['SOFT', 'HARD', 'MEDIUM', 'HARD']
                ]
                
                for pattern in balanced_patterns:
                    if all(compound in self.config.compounds for compound in pattern):
                        strategy = list(zip(pattern, stint_lengths))
                        if self._is_valid_strategy(strategy):
                            strategies.append(strategy)
        
        return strategies
    
    def _generate_conservative_strategies(self) -> List[List[Tuple[str, int]]]:
        """Generate conservative strategies with longer stints."""
        strategies = []
        
        if self.config.rain:
            return strategies
        
        # Conservative 2-stint strategies
        for pit_lap in range(max(15, self.config.total_laps // 2 - 5), 
                           min(self.config.total_laps - 8, self.config.total_laps // 2 + 8)):
            first_stint = pit_lap
            second_stint = self.config.total_laps - pit_lap
            
            if (first_stint >= self.config.min_stint_length and 
                second_stint >= self.config.min_stint_length):
                
                # Conservative compound choices
                conservative_combos = [
                    ('MEDIUM', 'HARD'),
                    ('HARD', 'MEDIUM'),
                    ('SOFT', 'HARD'),
                ]
                
                for combo in conservative_combos:
                    strategy = [(combo[0], first_stint), (combo[1], second_stint)]
                    if self._is_valid_strategy(strategy):
                        strategies.append(strategy)
        
        return strategies
    
    def _is_valid_strategy(self, strategy: List[Tuple[str, int]]) -> bool:
        """Validate if a strategy meets all constraints."""
        total_laps = sum(stint[1] for stint in strategy)
        
        # Check total laps
        if total_laps != self.config.total_laps:
            return False
        
        # Check minimum stint lengths
        for compound, length in strategy:
            if length < self.config.min_stint_length:
                return False
        
        # Check maximum stint lengths
        if self.config.max_stint_length:
            for compound, length in strategy:
                if length > self.config.max_stint_length:
                    return False
        
        # Check compound requirements for dry races
        if not self.config.rain:
            compounds_used = {stint[0] for stint in strategy}
            if len(compounds_used) < 2:
                return False
        
        # Check that all compounds are valid
        for compound, _ in strategy:
            if compound not in self.config.compounds:
                return False
        
        return True
    
    def _filter_and_deduplicate(self, strategies: List[List[Tuple[str, int]]]) -> List[List[Tuple[str, int]]]:
        """Remove duplicate and similar strategies."""
        unique_strategies = []
        seen_signatures: Set[str] = set()
        
        for strategy in strategies:
            # Create signature for deduplication
            signature = str(sorted(strategy))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_strategies.append(strategy)
        
        return unique_strategies


class SimplifiedStrategyEvaluator:
    """Simplified strategy evaluator focused on lap time predictions."""
    
    def __init__(self, model: Any, pit_stop_time: float = 22.0):
        self.model = model
        self.pit_stop_time = pit_stop_time
        self.feature_builder = EnhancedFeatureBuilder()
        
    def evaluate_strategy(
        self,
        strategy: List[Tuple[str, int]],
        driver: str,
        team: str,
        race: str,
        qual_time: float,
        start_pos: int,
        rain: bool,
        track_temp: float = 30.0,
        air_temp: float = 25.0
    ) -> StrategyResult:
        """
        Simplified strategy evaluation focused on lap time prediction.
        
        Returns:
            StrategyResult with basic analysis
        """
        try:
            # Build strategy dataframe
            strategy_df = self._build_dataframe(
                strategy, driver, team, race, qual_time, start_pos, rain, track_temp, air_temp
            )
            
            # Add enhanced features
            total_laps = sum(stint[1] for stint in strategy)
            strategy_df = self.feature_builder.add_enhanced_features(strategy_df, total_laps)
            
            # Predict lap times using the model
            predicted_times = self.model.predict(strategy_df)
            
            # Calculate pit stop penalties (adjustable pit stop time)
            num_pit_stops = len(strategy) - 1
            total_pit_time = num_pit_stops * self.pit_stop_time
            
            # Calculate total time (pure prediction model output + pit stops)
            total_time = predicted_times.sum() + total_pit_time
            average_lap_time = predicted_times.mean()
            
            # Add predictions to dataframe
            strategy_df['Predicted_Lap_Time'] = predicted_times
            
            return StrategyResult(
                strategy=strategy,
                total_time=total_time,
                average_lap_time=average_lap_time,
                pit_stop_time=self.pit_stop_time,
                dataframe=strategy_df
            )
            
        except Exception as e:
            print(f"Error evaluating strategy {strategy}: {e}")
            return StrategyResult(
                strategy=strategy,
                total_time=np.inf,
                average_lap_time=np.inf,
                pit_stop_time=self.pit_stop_time,
                dataframe=None
            )
    
    def _build_dataframe(
        self,
        strategy: List[Tuple[str, int]],
        driver: str,
        team: str,
        race: str,
        qual_time: float,
        start_pos: int,
        rain: bool,
        track_temp: float,
        air_temp: float
    ) -> pd.DataFrame:
        """Build dataframe for strategy evaluation."""
        rows = []
        current_lap = 1
        
        for stint_idx, (compound, stint_length) in enumerate(strategy):
            for lap_in_stint in range(1, stint_length + 1):
                # Calculate dynamic track temperature (varies throughout race)
                dynamic_track_temp = track_temp + np.sin(current_lap / 10) * 2
                
                row = {
                    'Driver': driver,
                    'Team': team,
                    'Compound': compound,
                    'FreshTyre': lap_in_stint == 1,
                    'PitLap': lap_in_stint == 1 and current_lap != 1,
                    'EventName': race,
                    'EventYear': 2025,
                    'TrackStatus': ['1'],
                    'LapTime_Qualifying': qual_time,
                    'TyreLife': lap_in_stint,
                    'LapNumber': current_lap,
                    'StartingPosition': start_pos,
                    'Rainfall': rain,
                    'AirTemp': air_temp,
                    'TrackTemp': dynamic_track_temp,
                    'Position': start_pos  # Simplified position modeling
                }
                rows.append(row)
                current_lap += 1
        
        return pd.DataFrame(rows)


def find_optimal_strategy(
    model: Any,
    driver: str,
    team: str,
    race: str,
    qual_time: float,
    start_pos: int,
    rain: bool,
    total_laps: int,
    num_stints: Optional[int] = None,
    track_temp: float = 30.0,
    air_temp: float = 25.0,
    pit_stop_time: float = 22.0
) -> Tuple[Optional[StrategyResult], List[StrategyResult]]:
    """
    Find optimal tire strategy using simplified approach focused on lap time prediction.
    
    Args:
        model: Trained prediction model
        driver: Driver code
        team: Team name
        race: Race name
        qual_time: Qualifying lap time
        start_pos: Starting grid position
        rain: Rain conditions
        total_laps: Total race laps
        num_stints: Fixed number of stints (optional)
        track_temp: Track temperature
        air_temp: Air temperature
        pit_stop_time: Pit stop penalty time in seconds
    
    Returns:
        Tuple of (best_strategy_result, top_strategies_list)
    """
    # Configure strategy generation
    compounds = ['INTERMEDIATE', 'WET'] if rain else ['SOFT', 'MEDIUM', 'HARD']
    
    config = StrategyConfig(
        total_laps=total_laps,
        compounds=compounds,
        rain=rain,
        fixed_stints=num_stints,
        max_strategies=2000,
        pit_stop_time=pit_stop_time
    )
    
    # Generate strategies
    print(f"Generating strategies for {total_laps} laps...")
    generator = SimplifiedStrategyGenerator(config)
    strategies = generator.generate_strategies()
    
    print(f"Evaluating {len(strategies)} strategies using lap time prediction model...")
    
    # Evaluate strategies
    evaluator = SimplifiedStrategyEvaluator(model, pit_stop_time)
    
    # Use ThreadPoolExecutor for better performance
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_strategy = {
            executor.submit(
                evaluator.evaluate_strategy,
                strategy, driver, team, race, qual_time, start_pos, rain, track_temp, air_temp
            ): strategy for strategy in strategies
        }
        
        for future in tqdm(as_completed(future_to_strategy), total=len(strategies), desc="Evaluating strategies"):
            result = future.result()
            if result.total_time != np.inf:
                results.append(result)
    
    if not results:
        print("No valid strategies found!")
        return None, []
    
    # Sort strategies by total time (purely based on model predictions + pit stops)
    results.sort(key=lambda x: x.total_time)
    
    # Get top strategies with variety in stint counts
    top_strategies = []
    seen_signatures: Set[str] = set()
    stint_counts = {}
    
    for result in results:
        strategy_signature = str(sorted(result.strategy))
        stint_count = len(result.strategy)
        
        # Ensure variety in stint counts (max 2 per stint count)
        if (strategy_signature not in seen_signatures and 
            len(top_strategies) < 8 and
            stint_counts.get(stint_count, 0) < 2):
            
            seen_signatures.add(strategy_signature)
            stint_counts[stint_count] = stint_counts.get(stint_count, 0) + 1
            top_strategies.append(result)
    
    # Fill remaining slots if needed
    if len(top_strategies) < 5:
        for result in results:
            strategy_signature = str(sorted(result.strategy))
            if strategy_signature not in seen_signatures and len(top_strategies) < 5:
                seen_signatures.add(strategy_signature)
                top_strategies.append(result)
    
    best_strategy = top_strategies[0] if top_strategies else None
    
    if best_strategy:
        print(f"Best strategy found (pit stop time: {pit_stop_time}s):")
        print(f"  Strategy: {' → '.join([f'{c}({l})' for c, l in best_strategy.strategy])}")
        print(f"  Total time: {best_strategy.total_time:.2f}s")
        print(f"  Avg lap time: {best_strategy.average_lap_time:.3f}s")
        print(f"  Stint count: {len(best_strategy.strategy)}")
    
    # Print summary of recommended strategies
    print(f"\nTop {len(top_strategies)} strategies:")
    for i, result in enumerate(top_strategies, 1):
        stint_summary = ' → '.join([f'{c}({l})' for c, l in result.strategy])
        print(f"  {i}. {stint_summary} — {result.total_time:.2f}s ({len(result.strategy)} stints)")
    
    return best_strategy, top_strategies


# Backward compatibility function
def find_best_strategy(
    model: Any,
    driver: str,
    team: str,
    race: str,
    qual_time: float,
    start_pos: int,
    rain: bool,
    total_laps: int,
    num_stints: Optional[int] = None,
    pit_stop_time: float = 22.0
) -> Tuple[Optional[List[Tuple[str, int]]], float, Optional[pd.DataFrame], List[Tuple[float, List[Tuple[str, int]], pd.DataFrame]]]:
    """
    Backward compatibility wrapper for the simplified strategy finder.
    """
    best_result, top_results = find_optimal_strategy(
        model, driver, team, race, qual_time, start_pos, rain, total_laps, num_stints, pit_stop_time=pit_stop_time
    )
    
    if not best_result:
        return None, np.inf, None, []
    
    # Convert to old format
    top_strategies = []
    for result in top_results:
        top_strategies.append((result.total_time, result.strategy, result.dataframe))
    
    return best_result.strategy, best_result.total_time, best_result.dataframe, top_strategies