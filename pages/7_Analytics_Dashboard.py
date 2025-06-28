import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.analytics import get_analytics_summary, track_page_view, track_engagement

# Configure page
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    # Track page view
    track_page_view("Analytics Dashboard")
    
    st.title("📊 Privacy-Compliant Analytics Dashboard")
    st.markdown("### Anonymous visitor insights and engagement metrics")
    
    # Privacy notice
    with st.expander("🔒 Privacy Notice", expanded=False):
        st.markdown("""
        **Privacy-First Analytics**
        - No personal information is collected
        - No IP addresses are stored
        - No tracking cookies are used
        - All data is anonymous and aggregated
        - Sessions are identified by temporary, non-identifying tokens
        - Compliant with privacy regulations (GDPR, CCPA)
        
        **What we track:**
        - Page views (anonymous)
        - User engagement with features
        - General usage patterns
        - Session duration (anonymous)
        """)
    
    # Get analytics data
    analytics_data = get_analytics_summary()
    
    # Overview metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Visits", analytics_data["total_visits"])
    
    with col2:
        st.metric("Total Page Views", analytics_data["total_page_views"])
    
    with col3:
        avg_pages = analytics_data["total_page_views"] / max(analytics_data["total_visits"], 1)
        st.metric("Pages per Visit", f"{avg_pages:.1f}")
    
    with col4:
        recent_visits = [x[1] for x in analytics_data["recent_activity"]]
        today_visits = recent_visits[0] if recent_visits else 0
        st.metric("Today's Visits", today_visits)
    
    # Recent activity chart
    st.subheader("📅 Recent Activity (Last 7 Days)")
    if analytics_data["recent_activity"]:
        dates, visits = zip(*analytics_data["recent_activity"])
        
        fig_activity = go.Figure(data=[
            go.Bar(x=list(reversed(dates)), y=list(reversed(visits)), 
                   marker_color='lightblue', name='Daily Visits')
        ])
        
        fig_activity.update_layout(
            title="Daily Visits Trend",
            xaxis_title="Date",
            yaxis_title="Visits",
            showlegend=False
        )
        
        st.plotly_chart(fig_activity, use_container_width=True)
    else:
        st.info("No recent activity data available.")
    
    # Top pages
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Most Popular Pages")
        if analytics_data["top_pages"]:
            top_pages_df = pd.DataFrame(analytics_data["top_pages"], 
                                      columns=["Page", "Views"])
            
            fig_pages = px.bar(top_pages_df, x="Views", y="Page", 
                             orientation='h', title="Page Views")
            fig_pages.update_layout(height=400)
            st.plotly_chart(fig_pages, use_container_width=True)
        else:
            st.info("No page view data available.")
    
    with col2:
        st.subheader("🎯 User Engagement")
        if analytics_data["engagement_data"]:
            # Aggregate engagement data
            total_engagement = {}
            for date_data in analytics_data["engagement_data"].values():
                for action, count in date_data.items():
                    total_engagement[action] = total_engagement.get(action, 0) + count
            
            if total_engagement:
                engagement_df = pd.DataFrame(list(total_engagement.items()), 
                                           columns=["Action", "Count"])
                
                fig_engagement = px.pie(engagement_df, values="Count", names="Action",
                                      title="Engagement by Action")
                fig_engagement.update_layout(height=400)
                st.plotly_chart(fig_engagement, use_container_width=True)
            else:
                st.info("No engagement data available.")
        else:
            st.info("No engagement data available.")
    
    # Detailed analytics
    with st.expander("📋 Detailed Analytics", expanded=False):
        st.subheader("Raw Analytics Data")
        
        # Recent activity table
        if analytics_data["recent_activity"]:
            st.write("**Recent Daily Visits:**")
            activity_df = pd.DataFrame(analytics_data["recent_activity"], 
                                     columns=["Date", "Visits"])
            st.dataframe(activity_df, use_container_width=True)
        
        # Page views table
        if analytics_data["top_pages"]:
            st.write("**Page Views:**")
            pages_df = pd.DataFrame(analytics_data["top_pages"], 
                                  columns=["Page", "Views"])
            st.dataframe(pages_df, use_container_width=True)
        
        # Engagement table
        if analytics_data["engagement_data"]:
            st.write("**Daily Engagement:**")
            for date, engagement in analytics_data["engagement_data"].items():
                st.write(f"**{date}:**")
                engagement_df = pd.DataFrame(list(engagement.items()), 
                                           columns=["Action", "Count"])
                st.dataframe(engagement_df, use_container_width=True)
    
    # Export functionality
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Analytics Summary"):
            track_engagement("export_analytics")
            
            # Create summary report
            summary_report = {
                "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_visits": analytics_data["total_visits"],
                "total_page_views": analytics_data["total_page_views"],
                "recent_activity": analytics_data["recent_activity"],
                "top_pages": analytics_data["top_pages"]
            }
            
            # Convert to JSON string for download
            import json
            json_str = json.dumps(summary_report, indent=2)
            
            st.download_button(
                label="Download Analytics Report (JSON)",
                data=json_str,
                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export Page Views CSV"):
            track_engagement("export_csv")
            
            if analytics_data["top_pages"]:
                pages_df = pd.DataFrame(analytics_data["top_pages"], 
                                      columns=["Page", "Views"])
                csv_str = pages_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Page Views (CSV)",
                    data=csv_str,
                    file_name=f"page_views_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No page view data to export.")
    
    # Real-time engagement test
    st.subheader("🧪 Test Analytics Tracking")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test Button Click"):
            track_engagement("test_button_click")
            st.success("Button click tracked!")
    
    with col2:
        if st.button("Test Feature Usage"):
            track_engagement("test_feature_usage")
            st.success("Feature usage tracked!")
    
    with col3:
        if st.button("Test Download"):
            track_engagement("test_download")
            st.success("Download action tracked!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Analytics Dashboard • Privacy-Compliant • No Personal Data Collected</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()