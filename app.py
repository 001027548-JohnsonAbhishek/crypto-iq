import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
import numpy as np
from utils.data_fetcher import CryptoDataFetcher
from utils.technical_analysis import TechnicalAnalyzer
from utils.analytics import track_page_view, track_engagement
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Crypto Analysis Platform",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'crypto_data' not in st.session_state:
    st.session_state.crypto_data = None
if 'selected_crypto' not in st.session_state:
    st.session_state.selected_crypto = 'BTC-USD'

def main():
    # Track page view
    track_page_view("Home")
    
    st.title("🚀 Comprehensive Cryptocurrency Analysis Platform")
    st.markdown("### Advanced ML Predictions, Sentiment Analysis & Portfolio Optimization")
    
    # Sidebar for crypto selection
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Popular cryptocurrencies
        popular_cryptos = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Binance Coin': 'BNB-USD',
            'Cardano': 'ADA-USD',
            'Solana': 'SOL-USD',
            'XRP': 'XRP-USD',
            'Polkadot': 'DOT-USD',
            'Dogecoin': 'DOGE-USD',
            'Avalanche': 'AVAX-USD',
            'Polygon': 'MATIC-USD'
        }
        
        # Crypto selection
        selected_name = st.selectbox(
            "Select Cryptocurrency",
            options=list(popular_cryptos.keys()),
            index=0
        )
        
        selected_symbol = popular_cryptos[selected_name]
        
        # Custom symbol input
        custom_symbol = st.text_input(
            "Or enter custom symbol (e.g., LINK-USD)",
            placeholder="LINK-USD"
        )
        
        if custom_symbol:
            selected_symbol = custom_symbol.upper()
            selected_name = custom_symbol.split('-')[0]
        
        st.session_state.selected_crypto = selected_symbol
        
        # Time range selection
        time_ranges = {
            '1 Month': 30,
            '3 Months': 90,
            '6 Months': 180,
            '1 Year': 365,
            '2 Years': 730,
            '5 Years': 1825
        }
        
        selected_range = st.selectbox(
            "Select Time Range",
            options=list(time_ranges.keys()),
            index=3
        )
        
        days = time_ranges[selected_range]
        
        # Fetch data button
        if st.button("🔄 Fetch Data", type="primary"):
            with st.spinner(f"Fetching data for {selected_name}..."):
                fetcher = CryptoDataFetcher()
                data = fetcher.get_crypto_data(selected_symbol, days)
                if data is not None and not data.empty:
                    st.session_state.crypto_data = data
                    st.success(f"✅ Data fetched successfully for {selected_name}")
                else:
                    st.error("❌ Failed to fetch data. Please check the symbol.")
    
    # Main content area
    if st.session_state.crypto_data is not None:
        data = st.session_state.crypto_data
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                label=f"{selected_name} Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col2:
            st.metric(
                label="24h Volume",
                value=f"${data['Volume'].iloc[-1]:,.0f}"
            )
        
        with col3:
            high_52w = data['High'].max()
            low_52w = data['Low'].min()
            st.metric(
                label="52W High",
                value=f"${high_52w:.2f}"
            )
        
        with col4:
            st.metric(
                label="52W Low",
                value=f"${low_52w:.2f}"
            )
        
        # Price chart
        st.subheader("📈 Price Chart")
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=selected_name
        ))
        
        fig.update_layout(
            title=f"{selected_name} Price Chart",
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader("📊 Volume Analysis")
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.8)'
        ))
        
        fig_volume.update_layout(
            title=f"{selected_name} Trading Volume",
            yaxis_title="Volume",
            xaxis_title="Date",
            height=300
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Data table
        st.subheader("📋 Recent Data")
        
        # Display last 10 rows
        display_data = data.tail(10).copy()
        display_data = display_data.round(4)
        st.dataframe(display_data, use_container_width=True)
        
        # Download section
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = data.to_csv()
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{selected_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = data.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"{selected_symbol}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        # Quick analysis
        st.subheader("🔍 Quick Analysis")
        
        analyzer = TechnicalAnalyzer()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            rsi = analyzer.calculate_rsi(data['Close'])
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 70:
                rsi_signal = "🔴 Overbought"
            elif current_rsi < 30:
                rsi_signal = "🟢 Oversold"
            else:
                rsi_signal = "🟡 Neutral"
            
            st.metric("RSI (14)", f"{current_rsi:.2f}", rsi_signal)
        
        with col2:
            # Moving averages
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            if current_price > ma_20 > ma_50:
                trend = "🟢 Bullish"
            elif current_price < ma_20 < ma_50:
                trend = "🔴 Bearish"
            else:
                trend = "🟡 Mixed"
            
            st.metric("Trend Analysis", trend)
        
        # Market sentiment
        st.subheader("💭 Market Sentiment")
        
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            # Volatility
            volatility = data['Close'].pct_change().std() * np.sqrt(365) * 100
            st.metric("Annualized Volatility", f"{volatility:.1f}%")
        
        with sentiment_col2:
            # Sharpe ratio approximation
            returns = data['Close'].pct_change().dropna()
            avg_return = returns.mean() * 365
            sharpe = avg_return / (returns.std() * np.sqrt(365))
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with sentiment_col3:
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        
    else:
        st.info("👈 Please select a cryptocurrency and click 'Fetch Data' to begin analysis")
        
        # Show available features
        st.subheader("🌟 Platform Features")
        
        features = [
            "📈 **Price Analysis** - Real-time price data with interactive charts",
            "🤖 **ML Predictions** - ARIMA, Prophet, and LSTM models for price forecasting",
            "📊 **Technical Analysis** - RSI, MACD, Bollinger Bands, and more indicators",
            "💭 **Sentiment Analysis** - Social media sentiment tracking",
            "💼 **Portfolio Optimization** - Modern portfolio theory implementation",
            "⚡ **Trading Signals** - Automated buy/sell signal generation",
            "📱 **Real-time Alerts** - Custom notification system",
            "💾 **Data Export** - CSV and JSON download capabilities"
        ]
        
        for feature in features:
            st.markdown(feature)
        
        st.info("Navigate to different pages using the sidebar to access advanced features!")

if __name__ == "__main__":
    main()
