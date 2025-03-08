# feedback_tracker.py
import pandas as pd
from datetime import datetime

class FeedbackTracker:
    def __init__(self, feedback_file="S:/LLM/user_feedback.csv"):
        self.feedback_file = feedback_file
        try:
            self.feedback_data = pd.read_csv(feedback_file)
        except FileNotFoundError:
            self.feedback_data = pd.DataFrame(columns=[
                'user_id', 'room_id', 'feedback', 'timestamp',
                'query_type', 'primary_factor'
            ])
            self.save_feedback_data()

    def add_feedback(self, user_id, room_id, feedback, query_type=None, primary_factor=None):
        new_feedback = {
            'user_id': user_id,
            'room_id': room_id,
            'feedback': feedback,  # 1 for positive, 0 for negative
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query_type': query_type,
            'primary_factor': primary_factor
        }
        self.feedback_data = pd.concat([self.feedback_data, 
                                      pd.DataFrame([new_feedback])], 
                                      ignore_index=True)
        self.save_feedback_data()

    def get_user_preferences(self, user_id):
        user_data = self.feedback_data[self.feedback_data['user_id'] == user_id]
        if len(user_data) == 0:
            return {
                'success_rate': None,
                'preferred_rooms': {},
                'primary_factors': {}
            }
        
        success_rate = float(user_data['feedback'].mean())
        preferred_rooms = user_data[user_data['feedback'] == 1]['room_id'].value_counts()
        
        return {
            'success_rate': success_rate,
            'preferred_rooms': preferred_rooms.to_dict(),
            'primary_factors': user_data['primary_factor'].value_counts().to_dict()
        }

    def save_feedback_data(self):
        self.feedback_data.to_csv(self.feedback_file, index=False)