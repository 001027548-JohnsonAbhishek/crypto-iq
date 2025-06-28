import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# Import our utilities
from utils.institutional_flow import InstitutionalFlowTracker
from utils.analytics import track_page_view, track_engagement

st.set_page_config(
    page_title="Institutional Flow Tracker",
    page_icon="🏛️",
    layout="wide"
)

def main():
    # Track page view
    track_page_view("Institutional Flow Tracker")
    
    st.title("🏛️ Institutional Investor Flow Tracker")
    st.markdown("### Monitor large-scale institutional trading patterns and whale activities")
    
    # Check if data exists in session state
    if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
        st.warning("⚠️ Please select a cryptocurrency and fetch data from the main page first.")
        if st.button("Go to Main Page"):
            st.switch_page("app.py")
        return
    
    symbol = st.session_state.get('selected_crypto', 'BTC')
    data = st.session_state.crypto_data
    
    # Initialize institutional flow tracker
    tracker = InstitutionalFlowTracker()
    
    # Configuration sidebar
    with st.sidebar:
        st.header("📊 Analysis Settings")
        
        analysis_timeframe = st.selectbox(
            "Analysis Timeframe:",
            ["7 days", "30 days", "90 days"],
            index=1
        )
        
        flow_threshold = st.slider(
            "Institutional Volume Threshold ($ millions):",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum transaction size to consider institutional"
        )
        
        show_advanced = st.checkbox("Show Advanced Metrics", value=False)
        
        if st.button("🔄 Refresh Analysis"):
            track_engagement("institutional_analysis_refresh", {"symbol": symbol})
            st.rerun()
    
    # Convert timeframe to days
    timeframe_map = {"7 days": 7, "30 days": 30, "90 days": 90}
    days = timeframe_map[analysis_timeframe]
    
    # Update tracker threshold
    tracker.large_tx_threshold = flow_threshold * 1000000
    
    # Main analysis
    with st.spinner("Analyzing institutional flows..."):
        try:
            # Get comprehensive institutional analysis
            institutional_analysis = tracker.get_comprehensive_institutional_analysis(symbol)
            
            if "error" in institutional_analysis:
                st.error(f"Analysis failed: {institutional_analysis['error']}")
                return
            
            # Display overall score and summary
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                activity_score = institutional_analysis.get('overall_institutional_activity_score', 0)
                st.metric(
                    "Institutional Activity Score",
                    f"{activity_score:.1f}/100",
                    delta=f"{'High' if activity_score >= 70 else 'Medium' if activity_score >= 40 else 'Low'} Activity"
                )
            
            with col2:
                sentiment_data = institutional_analysis.get('institutional_sentiment', {})
                sentiment = sentiment_data.get('institutional_sentiment', 'Neutral')
                st.metric(
                    "Institutional Sentiment",
                    sentiment,
                    delta=f"Confidence: {sentiment_data.get('institutional_confidence', 50):.1f}%"
                )
            
            with col3:
                st.info(institutional_analysis.get('summary', 'Analysis completed'))
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Exchange Flows", 
                "🐋 Whale Activity", 
                "🏢 Holdings Analysis", 
                "📊 ETF Flows", 
                "💭 Sentiment"
            ])
            
            with tab1:
                st.subheader("📈 Exchange Flow Analysis")
                
                exchange_data = institutional_analysis.get('exchange_flows', {})
                if "error" not in exchange_data:
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "24h Volume",
                            f"${exchange_data.get('total_volume_24h', 0):,.0f}",
                            help="Total trading volume in last 24 hours"
                        )
                    
                    with col2:
                        inst_percentage = exchange_data.get('institutional_volume_percentage', 0)
                        st.metric(
                            "Institutional Volume %",
                            f"{inst_percentage:.1f}%",
                            delta="High" if inst_percentage > 30 else "Normal"
                        )
                    
                    with col3:
                        inst_events = exchange_data.get('institutional_events_count', 0)
                        st.metric(
                            "Large Volume Events",
                            inst_events,
                            help="Number of institutional-size volume spikes"
                        )
                    
                    with col4:
                        recent_activity = exchange_data.get('recent_institutional_activity', 0)
                        st.metric(
                            "Recent Activity (7d)",
                            recent_activity,
                            delta="Active" if recent_activity > 5 else "Quiet"
                        )
                    
                    # Largest volume event details
                    if 'largest_volume_hour' in exchange_data:
                        largest_event = exchange_data['largest_volume_hour']
                        st.subheader("🔍 Largest Volume Event")
                        
                        event_col1, event_col2, event_col3 = st.columns(3)
                        
                        with event_col1:
                            st.metric("Volume", f"{largest_event.get('volume', 0):,.0f}")
                        
                        with event_col2:
                            st.metric("Timestamp", largest_event.get('timestamp', 'N/A'))
                        
                        with event_col3:
                            price_impact = largest_event.get('price_impact', 0)
                            st.metric("Price Impact", f"${price_impact:,.2f}")
                    
                    # Volume threshold visualization
                    if show_advanced:
                        st.subheader("📊 Volume Analysis")
                        
                        # Create volume threshold chart
                        fig = go.Figure()
                        
                        # Add average volume line
                        avg_vol = exchange_data.get('avg_volume', 0)
                        inst_threshold = exchange_data.get('institutional_threshold', 0)
                        
                        fig.add_hline(y=avg_vol, line_dash="dash", line_color="blue", 
                                     annotation_text="Average Volume")
                        fig.add_hline(y=inst_threshold, line_dash="dash", line_color="red", 
                                     annotation_text="Institutional Threshold")
                        
                        fig.update_layout(
                            title="Volume Threshold Analysis",
                            yaxis_title="Volume",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"Exchange flow analysis error: {exchange_data['error']}")
            
            with tab2:
                st.subheader("🐋 Whale Transaction Analysis")
                
                whale_data = institutional_analysis.get('whale_transactions', {})
                if "error" not in whale_data:
                    # Whale metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        whale_count = whale_data.get('whale_transactions_count', 0)
                        st.metric(
                            "Whale Transactions",
                            whale_count,
                            delta="High Activity" if whale_count > 10 else "Normal"
                        )
                    
                    with col2:
                        total_whale_vol = whale_data.get('total_whale_volume', 0)
                        st.metric(
                            "Total Whale Volume",
                            f"{total_whale_vol:,.0f}",
                            help="Combined volume from all whale transactions"
                        )
                    
                    with col3:
                        avg_whale_size = whale_data.get('avg_whale_transaction_size', 0)
                        st.metric(
                            "Avg Whale Size",
                            f"{avg_whale_size:,.0f}",
                            help="Average size of whale transactions"
                        )
                    
                    with col4:
                        flow_direction = whale_data.get('whale_flow_direction', 'Neutral')
                        st.metric(
                            "Flow Direction",
                            flow_direction.split(' ')[0],
                            delta=flow_direction.split(' ')[1] if ' ' in flow_direction else ''
                        )
                    
                    # Recent whale activity
                    recent_whales = whale_data.get('recent_whale_activity', [])
                    if recent_whales:
                        st.subheader("🕒 Recent Whale Transactions")
                        
                        whale_df = pd.DataFrame(recent_whales)
                        if not whale_df.empty:
                            # Format the dataframe for display
                            whale_df['estimated_usd_value'] = whale_df['estimated_usd_value'].apply(lambda x: f"${x:,.0f}")
                            whale_df['volume'] = whale_df['volume'].apply(lambda x: f"{x:,.0f}")
                            whale_df['price'] = whale_df['price'].apply(lambda x: f"${x:,.2f}")
                            whale_df['price_change'] = whale_df['price_change'].apply(lambda x: f"${x:+.2f}")
                            whale_df['impact_score'] = whale_df['impact_score'].apply(lambda x: f"{x:.2f}")
                            
                            st.dataframe(
                                whale_df[['timestamp', 'estimated_usd_value', 'volume', 'price', 'price_change', 'impact_score']],
                                column_config={
                                    'timestamp': 'Time',
                                    'estimated_usd_value': 'USD Value',
                                    'volume': 'Volume',
                                    'price': 'Price',
                                    'price_change': 'Price Change',
                                    'impact_score': 'Impact Score'
                                },
                                use_container_width=True
                            )
                        
                        # Whale impact visualization
                        if show_advanced and len(recent_whales) > 1:
                            st.subheader("📊 Whale Impact Analysis")
                            
                            fig = go.Figure()
                            
                            whale_times = [w['timestamp'] for w in recent_whales]
                            whale_impacts = [w['impact_score'] for w in recent_whales]
                            whale_values = [w['estimated_usd_value'] for w in recent_whales]
                            
                            fig.add_trace(go.Scatter(
                                x=whale_times,
                                y=whale_impacts,
                                mode='markers',
                                marker=dict(
                                    size=[min(max(v/1000000, 5), 50) for v in whale_values],
                                    color=whale_impacts,
                                    colorscale='RdYlGn',
                                    showscale=True
                                ),
                                text=[f"${v:,.0f}" for v in whale_values],
                                hovertemplate="Time: %{x}<br>Impact: %{y:.2f}<br>Value: %{text}<extra></extra>"
                            ))
                            
                            fig.update_layout(
                                title="Whale Transaction Impact Over Time",
                                xaxis_title="Time",
                                yaxis_title="Impact Score",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.info("No recent whale transactions detected in the analysis period.")
                
                else:
                    st.error(f"Whale analysis error: {whale_data['error']}")
            
            with tab3:
                st.subheader("🏢 Institutional Holdings Analysis")
                
                holdings_data = institutional_analysis.get('institutional_holdings', {})
                if "error" not in holdings_data:
                    # Holdings metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        vwap = holdings_data.get('vwap', 0)
                        current_price = holdings_data.get('current_price', 0)
                        vwap_diff = holdings_data.get('price_vs_vwap', 0)
                        
                        st.metric(
                            "VWAP",
                            f"${vwap:,.2f}",
                            delta=f"{vwap_diff:+.1f}% vs current"
                        )
                    
                    with col2:
                        strength_score = holdings_data.get('institutional_strength_score', 0)
                        st.metric(
                            "Institutional Strength",
                            f"{strength_score:.1f}/100",
                            delta="Strong" if strength_score > 70 else "Moderate" if strength_score > 40 else "Weak"
                        )
                    
                    with col3:
                        est_percentage = holdings_data.get('estimated_institutional_percentage', 0)
                        st.metric(
                            "Est. Institutional %",
                            f"{est_percentage:.1f}%",
                            help="Estimated percentage of institutional ownership"
                        )
                    
                    # Accumulation/Distribution analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        acc_trend = holdings_data.get('accumulation_distribution_trend', 'Neutral')
                        st.metric(
                            "Accumulation Trend",
                            acc_trend,
                            delta="📈" if acc_trend == "Positive" else "📉"
                        )
                    
                    with col2:
                        obv_trend = holdings_data.get('obv_trend', 'Neutral')
                        st.metric(
                            "OBV Trend",
                            obv_trend,
                            delta="💰" if obv_trend == "Accumulation" else "💸"
                        )
                    
                    # Accumulation and Distribution periods
                    if show_advanced:
                        acc_periods = holdings_data.get('accumulation_periods', [])
                        dist_periods = holdings_data.get('distribution_periods', [])
                        
                        if acc_periods or dist_periods:
                            st.subheader("📅 Historical Patterns")
                            
                            pattern_col1, pattern_col2 = st.columns(2)
                            
                            with pattern_col1:
                                if acc_periods:
                                    st.write("**Recent Accumulation Periods:**")
                                    for period in acc_periods[-3:]:
                                        st.write(f"• {period['start_date']} to {period['end_date']}")
                                        st.write(f"  OBV: {period['obv_change']:+.1f}%, Price: {period['price_change']:+.1f}%")
                                        st.write(f"  Strength: {period['strength']}")
                            
                            with pattern_col2:
                                if dist_periods:
                                    st.write("**Recent Distribution Periods:**")
                                    for period in dist_periods[-3:]:
                                        st.write(f"• {period['start_date']} to {period['end_date']}")
                                        st.write(f"  OBV: {period['obv_change']:+.1f}%, Price: {period['price_change']:+.1f}%")
                                        st.write(f"  Strength: {period['strength']}")
                
                else:
                    st.error(f"Holdings analysis error: {holdings_data['error']}")
            
            with tab4:
                st.subheader("📊 ETF Flow Analysis")
                
                etf_data = institutional_analysis.get('etf_flows', {})
                if "error" not in etf_data:
                    # ETF metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        etf_influence = etf_data.get('estimated_etf_influence', 0)
                        st.metric(
                            "ETF Influence",
                            f"{etf_influence:.3f}",
                            delta="High" if etf_influence > 0.5 else "Low"
                        )
                    
                    with col2:
                        consistency = etf_data.get('volume_consistency_score', 0)
                        st.metric(
                            "Volume Consistency",
                            f"{consistency:.3f}",
                            help="Higher values indicate more consistent institutional trading"
                        )
                    
                    with col3:
                        high_vol_days = etf_data.get('high_volume_trading_days', 0)
                        st.metric(
                            "High Volume Days",
                            high_vol_days,
                            help="Days with potential ETF activity"
                        )
                    
                    with col4:
                        flow_trend = etf_data.get('etf_flow_trend', 'Neutral')
                        st.metric(
                            "Flow Trend",
                            flow_trend,
                            delta="🟢" if flow_trend == "Inflow" else "🔴"
                        )
                    
                    # Potential ETF days
                    potential_days = etf_data.get('potential_etf_days', [])
                    if potential_days and show_advanced:
                        st.subheader("📅 Potential ETF Activity Days")
                        
                        etf_col1, etf_col2 = st.columns(2)
                        
                        with etf_col1:
                            st.write("**Recent High Volume Days:**")
                            for day in potential_days[-10:]:
                                st.write(f"• {day}")
                        
                        with etf_col2:
                            avg_volume = etf_data.get('avg_daily_volume', 0)
                            st.metric("Average Daily Volume", f"{avg_volume:,.0f}")
                
                else:
                    st.error(f"ETF analysis error: {etf_data['error']}")
            
            with tab5:
                st.subheader("💭 Institutional Sentiment Analysis")
                
                sentiment_data = institutional_analysis.get('institutional_sentiment', {})
                if "error" not in sentiment_data:
                    # Sentiment metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        sentiment = sentiment_data.get('institutional_sentiment', 'Neutral')
                        st.metric(
                            "Overall Sentiment",
                            sentiment,
                            delta="📈" if "Bullish" in sentiment else "📉" if "Bearish" in sentiment else "➡️"
                        )
                    
                    with col2:
                        confidence = sentiment_data.get('institutional_confidence', 0)
                        st.metric(
                            "Confidence Level",
                            f"{confidence:.1f}%",
                            delta="High" if confidence > 70 else "Medium" if confidence > 40 else "Low"
                        )
                    
                    with col3:
                        volatility = sentiment_data.get('volatility', 0)
                        st.metric(
                            "Volatility",
                            f"{volatility:.1f}%",
                            delta="High" if volatility > 0.5 else "Normal"
                        )
                    
                    with col4:
                        sharpe = sentiment_data.get('sharpe_ratio', 0)
                        st.metric(
                            "Sharpe Ratio",
                            f"{sharpe:.2f}",
                            delta="Good" if sharpe > 1 else "Fair" if sharpe > 0 else "Poor"
                        )
                    
                    # Additional sentiment indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        momentum = sentiment_data.get('momentum_signal', 'Neutral')
                        st.metric("Price Momentum", momentum)
                    
                    with col2:
                        volume_trend = sentiment_data.get('volume_trend', 'Stable')
                        st.metric("Volume Trend", volume_trend)
                    
                    if show_advanced:
                        stability_score = sentiment_data.get('price_stability_score', 0)
                        st.metric(
                            "Price Stability Score",
                            f"{stability_score:.3f}",
                            help="Higher values indicate more stable price action"
                        )
                
                else:
                    st.error(f"Sentiment analysis error: {sentiment_data['error']}")
            
            # Download section
            st.header("📥 Export Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export detailed analysis as JSON
                analysis_json = json.dumps(institutional_analysis, indent=2, default=str)
                st.download_button(
                    label="📄 Download Detailed Analysis (JSON)",
                    data=analysis_json,
                    file_name=f"{symbol}_institutional_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Export summary as CSV
                summary_data = {
                    'Metric': [
                        'Overall Activity Score',
                        'Institutional Sentiment',
                        'Institutional Confidence',
                        'Whale Transactions Count',
                        'Institutional Volume %',
                        'ETF Influence',
                        'Accumulation Trend',
                        'OBV Trend'
                    ],
                    'Value': [
                        f"{institutional_analysis.get('overall_institutional_activity_score', 0):.1f}/100",
                        sentiment_data.get('institutional_sentiment', 'N/A'),
                        f"{sentiment_data.get('institutional_confidence', 0):.1f}%",
                        whale_data.get('whale_transactions_count', 0),
                        f"{exchange_data.get('institutional_volume_percentage', 0):.1f}%",
                        f"{etf_data.get('estimated_etf_influence', 0):.3f}",
                        holdings_data.get('accumulation_distribution_trend', 'N/A'),
                        holdings_data.get('obv_trend', 'N/A')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Summary (CSV)",
                    data=summary_csv,
                    file_name=f"{symbol}_institutional_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            # Track successful analysis
            track_engagement("institutional_analysis_completed", {
                "symbol": symbol,
                "activity_score": activity_score,
                "timeframe": analysis_timeframe
            })
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Please try refreshing the analysis or selecting a different timeframe.")

if __name__ == "__main__":
    main()