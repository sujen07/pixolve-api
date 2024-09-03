from fastapi import HTTPException
from datetime import datetime, timedelta
from typing import Dict

class RateLimiter:
    def __init__(self):
        self.user_calls: Dict[str, Dict[str, int]] = {}

    def check_rate_limit(self, user_id: str, max_calls: int = 2):
        today = datetime.now().date()
        if user_id not in self.user_calls:
            self.user_calls[user_id] = {str(today): 1}
            return
        
        user_data = self.user_calls[user_id]
        if str(today) not in user_data:
            user_data[str(today)] = 1
        elif user_data[str(today)] < max_calls:
            user_data[str(today)] += 1
        else:
            raise HTTPException(status_code=429, detail="Daily API call limit exceeded")

        # Clean up old data
        for date in list(user_data.keys()):
            if datetime.strptime(date, "%Y-%m-%d").date() < today - timedelta(days=1):
                del user_data[date]

rate_limiter = RateLimiter()