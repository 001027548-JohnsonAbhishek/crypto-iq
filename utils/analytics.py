import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

class PrivacyCompliantAnalytics:
    """
    Privacy-compliant analytics that only tracks anonymous, aggregated data
    No personal information, IP addresses, or identifying data is collected
    """
    
    def __init__(self):
        self.analytics_file = "analytics_data.json"
        self.ensure_analytics_file()
    
    def ensure_analytics_file(self):
        """Create analytics file if it doesn't exist"""
        if not os.path.exists(self.analytics_file):
            initial_data = {
                "daily_visits": {},
                "page_views": {},
                "session_data": {},
                "user_engagement": {}
            }
            with open(self.analytics_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def get_session_id(self):
        """Generate anonymous session identifier"""
        if 'session_id' not in st.session_state:
            # Create anonymous session ID (no personal data)
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now().microsecond)) % 10000}"
        return st.session_state.session_id
    
    def track_page_view(self, page_name):
        """Track anonymous page views"""
        try:
            session_id = self.get_session_id()
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M')
            
            # Load existing data
            with open(self.analytics_file, 'r') as f:
                data = json.load(f)
            
            # Track daily visits (anonymous)
            if current_date not in data["daily_visits"]:
                data["daily_visits"][current_date] = 0
            
            # Only count unique sessions per day
            if f"{current_date}_{session_id}" not in data.get("session_data", {}):
                data["daily_visits"][current_date] += 1
                
            # Track page views
            if page_name not in data["page_views"]:
                data["page_views"][page_name] = 0
            data["page_views"][page_name] += 1
            
            # Track session data (anonymous)
            data["session_data"][f"{current_date}_{session_id}"] = {
                "date": current_date,
                "time": current_time,
                "last_page": page_name
            }
            
            # Save updated data
            with open(self.analytics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            # Fail silently to not disrupt user experience
            pass
    
    def track_engagement(self, action, details=None):
        """Track user engagement (anonymous)"""
        try:
            session_id = self.get_session_id()
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Load existing data
            with open(self.analytics_file, 'r') as f:
                data = json.load(f)
            
            # Track engagement
            if current_date not in data["user_engagement"]:
                data["user_engagement"][current_date] = {}
            
            if action not in data["user_engagement"][current_date]:
                data["user_engagement"][current_date][action] = 0
            
            data["user_engagement"][current_date][action] += 1
            
            # Save updated data
            with open(self.analytics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            # Fail silently
            pass
    
    def get_analytics_summary(self):
        """Get analytics summary for admin view"""
        try:
            with open(self.analytics_file, 'r') as f:
                data = json.load(f)
            
            # Calculate summary statistics
            total_visits = sum(data["daily_visits"].values())
            total_page_views = sum(data["page_views"].values())
            
            # Recent activity (last 7 days)
            recent_dates = []
            recent_visits = []
            
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                recent_dates.append(date)
                recent_visits.append(data["daily_visits"].get(date, 0))
            
            # Top pages
            top_pages = sorted(data["page_views"].items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total_visits": total_visits,
                "total_page_views": total_page_views,
                "recent_activity": list(zip(recent_dates, recent_visits)),
                "top_pages": top_pages,
                "engagement_data": data["user_engagement"]
            }
            
        except Exception as e:
            return {
                "total_visits": 0,
                "total_page_views": 0,
                "recent_activity": [],
                "top_pages": [],
                "engagement_data": {}
            }

# Global analytics instance
analytics = PrivacyCompliantAnalytics()

def track_page_view(page_name):
    """Helper function to track page views"""
    analytics.track_page_view(page_name)

def track_engagement(action, details=None):
    """Helper function to track engagement"""
    analytics.track_engagement(action, details)

def get_analytics_summary():
    """Helper function to get analytics summary"""
    return analytics.get_analytics_summary()