import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.sentiment_analysis import SentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sentiment Analysis", page_icon="💭", layout="wide")

def main():
    st.title("💭 Social Media & News Sentiment Analysis")
    st.markdown("Comprehensive sentiment analysis from Twitter, Reddit, and news sources")
    
    # Check if data exists in session state
    if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
        st.warning("⚠️ No data available. Please go back to the main page and fetch cryptocurrency data first.")
        return
    
    symbol = st.session_state.selected_crypto
    crypto_name = symbol.split('-')[0]  # Extract crypto name from symbol
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Sidebar for sentiment settings
    st.sidebar.header("🔧 Sentiment Settings")
    
    # Data sources selection
    st.sidebar.subheader("Data Sources")
    analyze_twitter = st.sidebar.checkbox("Twitter/X Analysis", value=False, 
                                          help="Requires Twitter API credentials")
    analyze_reddit = st.sidebar.checkbox("Reddit Analysis", value=False,
                                         help="Requires Reddit API credentials")
    analyze_news = st.sidebar.checkbox("News Analysis", value=False,
                                       help="Requires News API key")
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    max_tweets = st.sidebar.slider("Max Tweets", 50, 500, 100) if analyze_twitter else 100
    max_posts = st.sidebar.slider("Max Reddit Posts", 20, 200, 50) if analyze_reddit else 50
    
    # Subreddit selection for Reddit
    if analyze_reddit:
        subreddits = st.sidebar.multiselect(
            "Select Subreddits",
            ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin", "investing"],
            default=["CryptoCurrency", "Bitcoin"]
        )
    else:
        subreddits = ["CryptoCurrency", "Bitcoin"]
    
    # Manual sentiment input option
    st.sidebar.subheader("Manual Analysis")
    manual_text = st.sidebar.text_area(
        "Analyze Custom Text",
        placeholder="Enter text to analyze sentiment..."
    )
    
    if st.sidebar.button("🔍 Analyze Sentiment", type="primary"):
        sentiment_results = {}
        
        # Twitter Analysis
        if analyze_twitter:
            with st.spinner("Analyzing Twitter sentiment..."):
                twitter_sentiment = analyzer.get_twitter_sentiment(crypto_name, max_tweets)
                if twitter_sentiment:
                    sentiment_results['twitter'] = twitter_sentiment
                    st.success("✅ Twitter analysis completed")
                else:
                    st.warning("⚠️ Twitter analysis failed - check API credentials")
        
        # Reddit Analysis  
        if analyze_reddit:
            with st.spinner("Analyzing Reddit sentiment..."):
                reddit_sentiment = analyzer.get_reddit_sentiment(crypto_name, subreddits, max_posts)
                if reddit_sentiment:
                    sentiment_results['reddit'] = reddit_sentiment
                    st.success("✅ Reddit analysis completed")
                else:
                    st.warning("⚠️ Reddit analysis failed - check API credentials")
        
        # News Analysis
        if analyze_news:
            with st.spinner("Analyzing news sentiment..."):
                news_sentiment = analyzer.get_news_sentiment(crypto_name)
                if news_sentiment:
                    sentiment_results['news'] = news_sentiment
                    st.success("✅ News analysis completed")
                else:
                    st.warning("⚠️ News analysis failed - check API key")
        
        # Calculate overall sentiment if we have results
        if sentiment_results:
            overall_sentiment = analyzer.calculate_overall_sentiment(sentiment_results)
            if overall_sentiment:
                sentiment_results['overall'] = overall_sentiment
        
        # Store results in session state
        st.session_state.sentiment_results = sentiment_results
    
    # Manual text analysis
    if manual_text and st.sidebar.button("Analyze Text"):
        with st.spinner("Analyzing text sentiment..."):
            text_sentiment = analyzer.analyze_text_sentiment(manual_text)
            if text_sentiment:
                st.session_state.manual_sentiment = text_sentiment
                st.success("✅ Text analysis completed")
    
    # Display sentiment results
    if 'sentiment_results' in st.session_state and st.session_state.sentiment_results:
        results = st.session_state.sentiment_results
        
        # Overall Sentiment Dashboard
        if 'overall' in results:
            st.subheader("🎯 Overall Sentiment Dashboard")
            
            overall = results['overall']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment_emoji = "😊" if overall['label'] == 'Positive' else "😞" if overall['label'] == 'Negative' else "😐"
                st.metric("Overall Sentiment", f"{sentiment_emoji} {overall['label']}")
            
            with col2:
                confidence_color = "🟢" if overall['confidence'] > 0.5 else "🟡" if overall['confidence'] > 0.2 else "🔴"
                st.metric("Confidence", f"{confidence_color} {overall['confidence']:.2f}")
            
            with col3:
                st.metric("TextBlob Score", f"{overall['textblob_polarity']:+.2f}")
            
            with col4:
                st.metric("VADER Score", f"{overall['vader_compound']:+.2f}")
            
            # Sentiment gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = overall['vader_compound'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, -0.05], 'color': "orange"},
                        {'range': [-0.05, 0.05], 'color': "yellow"},
                        {'range': [0.05, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Individual Source Analysis
        st.subheader("📊 Source-Specific Analysis")
        
        # Twitter Analysis
        if 'twitter' in results:
            st.write("**🐦 Twitter Analysis**")
            
            twitter_data = results['twitter']
            agg_sentiment = twitter_data['aggregate_sentiment']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tweets Analyzed", agg_sentiment['total_tweets'])
            
            with col2:
                twitter_label = "Positive" if agg_sentiment['vader_compound'] > 0.05 else \
                               "Negative" if agg_sentiment['vader_compound'] < -0.05 else "Neutral"
                twitter_emoji = "😊" if twitter_label == 'Positive' else "😞" if twitter_label == 'Negative' else "😐"
                st.metric("Twitter Sentiment", f"{twitter_emoji} {twitter_label}")
            
            with col3:
                st.metric("VADER Score", f"{agg_sentiment['vader_compound']:+.2f}")
            
            # Twitter sentiment breakdown
            if 'individual_tweets' in twitter_data:
                tweets = twitter_data['individual_tweets']
                
                # Create sentiment distribution
                sentiment_scores = [tweet['sentiment']['vader_compound'] for tweet in tweets]
                
                fig_twitter_dist = go.Figure(data=[go.Histogram(
                    x=sentiment_scores,
                    nbinsx=20,
                    name="Tweet Sentiment Distribution"
                )])
                
                fig_twitter_dist.update_layout(
                    title="Twitter Sentiment Distribution",
                    xaxis_title="Sentiment Score",
                    yaxis_title="Number of Tweets",
                    height=300
                )
                
                st.plotly_chart(fig_twitter_dist, use_container_width=True)
                
                # Show sample tweets
                with st.expander("📱 Sample Tweets", expanded=False):
                    for i, tweet in enumerate(tweets[:5]):  # Show first 5 tweets
                        sentiment_color = "🟢" if tweet['sentiment']['vader_compound'] > 0.05 else \
                                        "🔴" if tweet['sentiment']['vader_compound'] < -0.05 else "🟡"
                        st.write(f"{sentiment_color} **Score: {tweet['sentiment']['vader_compound']:+.2f}**")
                        st.write(f"_{tweet['text'][:200]}..._" if len(tweet['text']) > 200 else f"_{tweet['text']}_")
                        st.write("---")
        
        # Reddit Analysis
        if 'reddit' in results:
            st.write("**🤖 Reddit Analysis**")
            
            reddit_data = results['reddit']
            agg_sentiment = reddit_data['aggregate_sentiment']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Posts Analyzed", agg_sentiment['total_posts'])
            
            with col2:
                reddit_label = "Positive" if agg_sentiment['vader_compound'] > 0.05 else \
                              "Negative" if agg_sentiment['vader_compound'] < -0.05 else "Neutral"
                reddit_emoji = "😊" if reddit_label == 'Positive' else "😞" if reddit_label == 'Negative' else "😐"
                st.metric("Reddit Sentiment", f"{reddit_emoji} {reddit_label}")
            
            with col3:
                st.metric("VADER Score", f"{agg_sentiment['vader_compound']:+.2f}")
            
            # Reddit sentiment by subreddit
            if 'individual_posts' in reddit_data:
                posts = reddit_data['individual_posts']
                
                # Group by subreddit
                subreddit_sentiment = {}
                for post in posts:
                    subreddit = post['subreddit']
                    if subreddit not in subreddit_sentiment:
                        subreddit_sentiment[subreddit] = []
                    subreddit_sentiment[subreddit].append(post['sentiment']['vader_compound'])
                
                # Create subreddit comparison chart
                if len(subreddit_sentiment) > 1:
                    subreddits_list = list(subreddit_sentiment.keys())
                    avg_sentiments = [np.mean(subreddit_sentiment[sub]) for sub in subreddits_list]
                    
                    fig_reddit_sub = go.Figure(data=[
                        go.Bar(x=subreddits_list, y=avg_sentiments, name="Average Sentiment by Subreddit")
                    ])
                    
                    fig_reddit_sub.update_layout(
                        title="Reddit Sentiment by Subreddit",
                        xaxis_title="Subreddit",
                        yaxis_title="Average Sentiment Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig_reddit_sub, use_container_width=True)
                
                # Show sample posts
                with st.expander("📝 Sample Reddit Posts", expanded=False):
                    for i, post in enumerate(posts[:3]):  # Show first 3 posts
                        sentiment_color = "🟢" if post['sentiment']['vader_compound'] > 0.05 else \
                                        "🔴" if post['sentiment']['vader_compound'] < -0.05 else "🟡"
                        st.write(f"{sentiment_color} **r/{post['subreddit']} - Score: {post['sentiment']['vader_compound']:+.2f}**")
                        st.write(f"**{post['title']}**")
                        if post['text']:
                            preview_text = post['text'][:300] + "..." if len(post['text']) > 300 else post['text']
                            st.write(preview_text)
                        st.write(f"👍 Score: {post['score']}")
                        st.write("---")
        
        # News Analysis
        if 'news' in results:
            st.write("**📰 News Analysis**")
            
            news_data = results['news']
            agg_sentiment = news_data['aggregate_sentiment']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Articles Analyzed", agg_sentiment['total_articles'])
            
            with col2:
                news_label = "Positive" if agg_sentiment['vader_compound'] > 0.05 else \
                            "Negative" if agg_sentiment['vader_compound'] < -0.05 else "Neutral"
                news_emoji = "😊" if news_label == 'Positive' else "😞" if news_label == 'Negative' else "😐"
                st.metric("News Sentiment", f"{news_emoji} {news_label}")
            
            with col3:
                st.metric("VADER Score", f"{agg_sentiment['vader_compound']:+.2f}")
            
            # News sentiment timeline
            if 'individual_articles' in news_data:
                articles = news_data['individual_articles']
                
                # Create timeline chart
                article_dates = []
                article_sentiments = []
                
                for article in articles:
                    try:
                        date = pd.to_datetime(article['published_at'])
                        article_dates.append(date)
                        article_sentiments.append(article['sentiment']['vader_compound'])
                    except:
                        continue
                
                if article_dates:
                    fig_news_timeline = go.Figure(data=[
                        go.Scatter(x=article_dates, y=article_sentiments, mode='markers+lines',
                                 name="News Sentiment Timeline",
                                 marker=dict(
                                     size=8,
                                     color=article_sentiments,
                                     colorscale='RdYlGn',
                                     showscale=True,
                                     colorbar=dict(title="Sentiment Score")
                                 ))
                    ])
                    
                    fig_news_timeline.update_layout(
                        title="News Sentiment Over Time",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig_news_timeline, use_container_width=True)
                
                # Show sample articles
                with st.expander("📰 Sample News Articles", expanded=False):
                    for i, article in enumerate(articles[:3]):  # Show first 3 articles
                        sentiment_color = "🟢" if article['sentiment']['vader_compound'] > 0.05 else \
                                        "🔴" if article['sentiment']['vader_compound'] < -0.05 else "🟡"
                        st.write(f"{sentiment_color} **Score: {article['sentiment']['vader_compound']:+.2f}**")
                        st.write(f"**{article['title']}**")
                        st.write(f"_{article['description']}_")
                        st.write(f"Source: {article['source']} | {article['published_at']}")
                        if article.get('url'):
                            st.write(f"[Read more]({article['url']})")
                        st.write("---")
        
        # Comparative Analysis
        if len([k for k in results.keys() if k != 'overall']) > 1:
            st.subheader("📈 Comparative Sentiment Analysis")
            
            sources = []
            sentiments = []
            
            for source, data in results.items():
                if source != 'overall' and 'aggregate_sentiment' in data:
                    sources.append(source.title())
                    sentiments.append(data['aggregate_sentiment']['vader_compound'])
            
            if sources:
                fig_comparison = go.Figure(data=[
                    go.Bar(x=sources, y=sentiments, name="Sentiment by Source",
                           marker_color=['green' if s > 0.05 else 'red' if s < -0.05 else 'yellow' for s in sentiments])
                ])
                
                fig_comparison.add_hline(y=0, line_dash="dash", line_color="black")
                fig_comparison.add_hline(y=0.05, line_dash="dot", line_color="green", 
                                       annotation_text="Positive Threshold")
                fig_comparison.add_hline(y=-0.05, line_dash="dot", line_color="red", 
                                       annotation_text="Negative Threshold")
                
                fig_comparison.update_layout(
                    title="Sentiment Comparison Across Sources",
                    xaxis_title="Source",
                    yaxis_title="Sentiment Score",
                    height=400
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Manual text analysis results
    if 'manual_sentiment' in st.session_state:
        st.subheader("📝 Manual Text Analysis")
        
        manual_result = st.session_state.manual_sentiment
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("TextBlob Polarity", f"{manual_result['textblob_polarity']:+.2f}")
        
        with col2:
            st.metric("TextBlob Subjectivity", f"{manual_result['textblob_subjectivity']:.2f}")
        
        with col3:
            st.metric("VADER Compound", f"{manual_result['vader_compound']:+.2f}")
        
        with col4:
            vader_label = "Positive" if manual_result['vader_compound'] > 0.05 else \
                         "Negative" if manual_result['vader_compound'] < -0.05 else "Neutral"
            st.metric("Classification", vader_label)
        
        # VADER component breakdown
        st.write("**VADER Sentiment Breakdown:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive", f"{manual_result['vader_positive']:.2f}")
        
        with col2:
            st.metric("Neutral", f"{manual_result['vader_neutral']:.2f}")
        
        with col3:
            st.metric("Negative", f"{manual_result['vader_negative']:.2f}")
    
    # Fear & Greed Index
    st.subheader("😨 Fear & Greed Index")
    
    if st.button("📊 Fetch Fear & Greed Index"):
        with st.spinner("Fetching Fear & Greed Index..."):
            fng_data = analyzer.get_fear_greed_index()
            
            if fng_data:
                current_fng = fng_data[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fng_value = int(current_fng['value'])
                    st.metric("Fear & Greed Index", fng_value)
                
                with col2:
                    st.metric("Classification", current_fng['value_classification'])
                
                with col3:
                    st.metric("Date", current_fng['timestamp'][:10])
                
                # Fear & Greed Index gauge
                fig_fng = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fng_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fear & Greed Index"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "red"},
                            {'range': [25, 45], 'color': "orange"},
                            {'range': [45, 55], 'color': "yellow"},
                            {'range': [55, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig_fng.update_layout(height=300)
                st.plotly_chart(fig_fng, use_container_width=True)
                
                # Historical Fear & Greed
                if len(fng_data) > 1:
                    dates = []
                    values = []
                    for item in fng_data:
                        try:
                            # Handle both unix timestamp and string dates
                            timestamp = item['timestamp']
                            if isinstance(timestamp, (int, float)) or timestamp.isdigit():
                                dates.append(pd.to_datetime(int(timestamp), unit='s'))
                            else:
                                dates.append(pd.to_datetime(timestamp))
                            values.append(int(item['value']))
                        except (ValueError, OverflowError):
                            # Skip invalid timestamps and their corresponding values
                            continue
                    
                    fig_fng_hist = go.Figure(data=[
                        go.Scatter(x=dates, y=values, mode='lines+markers',
                                 name="Fear & Greed Index",
                                 line=dict(width=3))
                    ])
                    
                    fig_fng_hist.add_hline(y=50, line_dash="dash", line_color="gray", 
                                          annotation_text="Neutral (50)")
                    fig_fng_hist.add_hline(y=25, line_dash="dot", line_color="red", 
                                          annotation_text="Extreme Fear (25)")
                    fig_fng_hist.add_hline(y=75, line_dash="dot", line_color="green", 
                                          annotation_text="Extreme Greed (75)")
                    
                    fig_fng_hist.update_layout(
                        title="Fear & Greed Index History",
                        xaxis_title="Date",
                        yaxis_title="Index Value",
                        height=400
                    )
                    
                    st.plotly_chart(fig_fng_hist, use_container_width=True)
            else:
                st.error("❌ Failed to fetch Fear & Greed Index")
    
    # Sentiment Analysis Guide
    if not any(['sentiment_results' in st.session_state, 'manual_sentiment' in st.session_state]):
        st.info("👆 Configure your sentiment analysis settings and click 'Analyze Sentiment' to start")
        
        st.subheader("📚 Sentiment Analysis Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Understanding Sentiment Scores**")
            st.write("• **Positive (+0.05 to +1.0)**: Bullish sentiment, optimistic outlook")
            st.write("• **Neutral (-0.05 to +0.05)**: Mixed or unclear sentiment")
            st.write("• **Negative (-1.0 to -0.05)**: Bearish sentiment, pessimistic outlook")
            
            st.write("**VADER vs TextBlob**")
            st.write("• **VADER**: Better for social media, handles emojis and slang")
            st.write("• **TextBlob**: More formal text analysis, good for news articles")
        
        with col2:
            st.write("**Data Sources**")
            st.write("• **Twitter/X**: Real-time public opinion and reactions")
            st.write("• **Reddit**: In-depth discussions and community sentiment")
            st.write("• **News**: Professional analysis and market coverage")
            
            st.write("**API Requirements**")
            st.write("• Twitter: Bearer Token")
            st.write("• Reddit: Client ID, Client Secret")
            st.write("• News: News API Key")
    
    # Export sentiment data
    if 'sentiment_results' in st.session_state:
        st.subheader("💾 Export Sentiment Data")
        
        # Prepare export data
        export_data = []
        results = st.session_state.sentiment_results
        
        for source, data in results.items():
            if source != 'overall' and 'aggregate_sentiment' in data:
                export_data.append({
                    'source': source,
                    'total_items': data['aggregate_sentiment'].get('total_tweets', 
                                                                  data['aggregate_sentiment'].get('total_posts',
                                                                                                  data['aggregate_sentiment'].get('total_articles', 0))),
                    'textblob_polarity': data['aggregate_sentiment']['textblob_polarity'],
                    'vader_compound': data['aggregate_sentiment']['vader_compound'],
                    'vader_positive': data['aggregate_sentiment']['vader_positive'],
                    'vader_negative': data['aggregate_sentiment']['vader_negative'],
                    'vader_neutral': data['aggregate_sentiment']['vader_neutral']
                })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sentiment CSV",
                    data=csv_data,
                    file_name=f"{crypto_name}_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = export_df.to_json(orient='records')
                st.download_button(
                    label="📥 Download Sentiment JSON",
                    data=json_data,
                    file_name=f"{crypto_name}_sentiment_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
