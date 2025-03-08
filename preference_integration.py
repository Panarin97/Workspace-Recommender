import pandas as pd
import numpy as np
from datetime import datetime, timedelta
    

class PreferenceAdjuster:
    def __init__(self, preferences_file="user_preferences.csv", learning_rate=0.2):
        self.preferences_file = preferences_file
        self.learning_rate = learning_rate
        self.user_preferences = pd.read_csv(preferences_file)
        
        # Sensitivity levels and their numeric values
        self.sensitivity_scales = {
            'temperature': ['very_cold_sensitive', 'cold_sensitive', 'slightly_cold_sensitive', 
                          'normal', 'slightly_heat_sensitive', 'heat_sensitive', 'very_heat_sensitive'],
            'humidity': ['very_dry_sensitive', 'dry_sensitive', 'slightly_dry_sensitive', 
                        'normal', 'slightly_humid_sensitive', 'humid_sensitive', 'very_humid_sensitive'],
            'co2': ['very_sensitive', 'sensitive', 'somewhat_sensitive', 
                   'normal', 'slightly_tolerant', 'tolerant', 'very_tolerant'],
            'light': ['extreme_light_sensitive', 'very_light_sensitive', 'light_sensitive', 
                     'slightly_light_sensitive', 'normal', 'slightly_dark_sensitive', 'dark_sensitive', 
                     'very_dark_sensitive', 'extreme_dark_sensitive'],
            'occupancy': ['extreme_quiet', 'very_quiet', 'quiet', 'somewhat_quiet', 
                         'normal', 'somewhat_busy', 'busy', 'very_busy', 'extreme_busy']
        }

    def adjust_preference(self, user_id, room_conditions, is_positive):
        """Adjust user preferences based on feedback for a room"""
        if user_id not in self.user_preferences['user_id'].values:
            return False, {}
            
        user_idx = self.user_preferences['user_id'] == user_id
        current_preferences = self.user_preferences.loc[user_idx].iloc[0]

        # For each sensor type, adjust preference toward or away from room conditions
        adjustments = {}
        for sensor_type in ['temperature', 'humidity', 'co2', 'light', 'occupancy']:
            current_level = current_preferences[sensor_type]
            current_idx = self.sensitivity_scales[sensor_type].index(current_level)
            
            # Determine direction of adjustment
            if is_positive:
                # Move preference toward room conditions
                if room_conditions[sensor_type] > 0:  # Room is on the "high" end
                    new_idx = max(current_idx - 1, 0)  # Move toward less sensitive
                else:
                    new_idx = min(current_idx + 1, len(self.sensitivity_scales[sensor_type]) - 1)
            else:
                # Move preference away from room conditions
                if room_conditions[sensor_type] > 0:
                    new_idx = min(current_idx + 1, len(self.sensitivity_scales[sensor_type]) - 1)
                else:
                    new_idx = max(current_idx - 1, 0)
            
            new_level = self.sensitivity_scales[sensor_type][new_idx]
            if new_level != current_level:
                adjustments[sensor_type] = {
                    'from': current_level,
                    'to': new_level
                }
        
        # Apply adjustments if any were made
        if adjustments:
                for sensor_type, change in adjustments.items():
                    self.user_preferences.loc[user_idx, sensor_type] = change['to']
                self.user_preferences.to_csv(self.preferences_file, index=False)
                return True, adjustments
            
        return False, {}

    def analyze_room_conditions(self, room_data, room_id):
        """Analyze room conditions relative to normal ranges"""
        room = room_data[room_data['Location'] == room_id].iloc[0]
        
        return {
            'temperature': (room['temperature_mean'] - 22) / 4,  # Normalized around 22°C
            'humidity': (room['humidity_mean'] - 40) / 20,  # Normalized around 40%
            'co2': (room['co2_mean'] - 800) / 400,  # Normalized around 800ppm
            'light': (room['light_mean'] - 300) / 200,  # Normalized around 300 lux
            'occupancy': (room['pir_mean'] - 3) / 3  # Normalized around 3
        }


preference_adjuster = PreferenceAdjuster()


def show_preference_history(app, user_id, feedback_tracker):
    """Display preference history in the Streamlit app."""
    preferences = preference_adjuster.user_preferences
    user_prefs = preferences[preferences['user_id'] == user_id].iloc[0]
    
    app.subheader("Your Preference Profile")
    for category in ['temperature', 'humidity', 'co2', 'light', 'occupancy']:
        app.write(f"{category.title()}: {user_prefs[category]}")
    
    # Get feedback history from feedback tracker
    history = feedback_tracker.get_user_preferences(user_id)
    if history and history['success_rate'] is not None:
        app.write(f"Positive feedback rate: {history['success_rate']*100:.1f}%")