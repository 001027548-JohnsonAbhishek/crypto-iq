import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_fetcher import CryptoDataFetcher
from utils.technical_analysis import TechnicalAnalyzer

st.set_page_config(page_title="Price Analysis", page_icon="📈", layout="wide")

def main():
    st.title("📈 Advanced Price Analysis")
    st.markdown("Comprehensive cryptocurrency price analysis with technical indicators")
    
    # Check if data exists in session state
    if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
        st.warning("⚠️ No data available. Please go back to the main page and fetch cryptocurrency data first.")
        return
    
    data = st.session_state.crypto_data
    symbol = st.session_state.selected_crypto
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line", "Area", "OHLC"]
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe Analysis",
            ["Daily", "Weekly", "Monthly"]
        )
        
    with col3:
        show_volume = st.checkbox("Show Volume", value=True)
    
    # Technical indicators selection
    st.subheader("📊 Technical Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_ma = st.checkbox("Moving Averages", value=True)
        show_ema = st.checkbox("Exponential MA")
        
    with col2:
        show_bb = st.checkbox("Bollinger Bands")
        show_rsi = st.checkbox("RSI", value=True)
        
    with col3:
        show_macd = st.checkbox("MACD")
        show_stoch = st.checkbox("Stochastic")
        
    with col4:
        show_support_resistance = st.checkbox("Support/Resistance")
        show_fibonacci = st.checkbox("Fibonacci Levels")
    
    # Initialize technical analyzer
    analyzer = TechnicalAnalyzer()
    
    # Create main price chart
    fig = go.Figure()
    
    # Add price data based on chart type
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff6b6b'
        ))
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='#1f77b4', width=2)
        ))
    elif chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            fill='tonexty',
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='#1f77b4'),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ))
    
    # Add technical indicators
    if show_ma:
        ma_periods = [20, 50, 100]
        colors = ['orange', 'red', 'purple']
        for period, color in zip(ma_periods, colors):
            ma = data['Close'].rolling(period).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma,
                mode='lines',
                name=f'MA{period}',
                line=dict(color=color, width=1)
            ))
    
    if show_ema:
        ema_periods = [12, 26]
        colors = ['yellow', 'cyan']
        for period, color in zip(ema_periods, colors):
            ema = data['Close'].ewm(span=period).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ema,
                mode='lines',
                name=f'EMA{period}',
                line=dict(color=color, width=1, dash='dash')
            ))
    
    if show_bb:
        bb = analyzer.calculate_bollinger_bands(data['Close'])
        if bb is not None and not bb.empty:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=bb['Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=bb['Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=True
            ))
    
    if show_support_resistance:
        # Simple support/resistance levels
        recent_high = data['High'].tail(50).max()
        recent_low = data['Low'].tail(50).min()
        
        fig.add_hline(y=recent_high, line_dash="dash", line_color="red", 
                     annotation_text="Resistance")
        fig.add_hline(y=recent_low, line_dash="dash", line_color="green", 
                     annotation_text="Support")
    
    if show_fibonacci:
        # Fibonacci retracements
        period_high = data['High'].max()
        period_low = data['Low'].min()
        
        fib_levels = analyzer.calculate_fibonacci_retracement(period_high, period_low)
        if fib_levels:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
            for i, (level, price) in enumerate(fib_levels.items()):
                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color=colors[i % len(colors)],
                    annotation_text=f"Fib {level}",
                    annotation_position="bottom right"
                )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Analysis",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    if show_volume:
        st.subheader("📊 Volume Analysis")
        
        fig_volume = go.Figure()
        
        # Color volume bars based on price change
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        
        fig_volume.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ))
        
        # Add volume moving average
        volume_ma = data['Volume'].rolling(20).mean()
        fig_volume.add_trace(go.Scatter(
            x=data.index,
            y=volume_ma,
            mode='lines',
            name='Volume MA(20)',
            line=dict(color='orange', width=2)
        ))
        
        fig_volume.update_layout(
            title="Trading Volume",
            yaxis_title="Volume",
            xaxis_title="Date",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Technical indicators subplots
    indicators_to_show = []
    if show_rsi:
        indicators_to_show.append('RSI')
    if show_macd:
        indicators_to_show.append('MACD')
    if show_stoch:
        indicators_to_show.append('Stochastic')
    
    if indicators_to_show:
        st.subheader("📈 Technical Indicators")
        
        for indicator in indicators_to_show:
            if indicator == 'RSI':
                rsi = analyzer.calculate_rsi(data['Close'])
                if rsi is not None and not rsi.empty:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=data.index,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # Add RSI levels
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                     annotation_text="Oversold (30)")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                     annotation_text="Midline")
                    
                    fig_rsi.update_layout(
                        title="RSI (Relative Strength Index)",
                        yaxis_title="RSI",
                        yaxis=dict(range=[0, 100]),
                        height=300
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
            
            elif indicator == 'MACD':
                macd = analyzer.calculate_macd(data['Close'])
                if macd is not None and not macd.empty:
                    fig_macd = go.Figure()
                    
                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=macd['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=macd['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_macd.add_trace(go.Bar(
                        x=data.index,
                        y=macd['Histogram'],
                        name='Histogram',
                        marker_color='gray',
                        opacity=0.6
                    ))
                    
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    fig_macd.update_layout(
                        title="MACD (Moving Average Convergence Divergence)",
                        yaxis_title="MACD",
                        height=300
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
            
            elif indicator == 'Stochastic':
                stoch = analyzer.calculate_stochastic(data['High'], data['Low'], data['Close'])
                if stoch is not None and not stoch.empty:
                    fig_stoch = go.Figure()
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=data.index,
                        y=stoch['%K'],
                        mode='lines',
                        name='%K',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=data.index,
                        y=stoch['%D'],
                        mode='lines',
                        name='%D',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add stochastic levels
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", 
                                       annotation_text="Overbought (80)")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", 
                                       annotation_text="Oversold (20)")
                    
                    fig_stoch.update_layout(
                        title="Stochastic Oscillator",
                        yaxis_title="Stochastic",
                        yaxis=dict(range=[0, 100]),
                        height=300
                    )
                    
                    st.plotly_chart(fig_stoch, use_container_width=True)
    
    # Price statistics
    st.subheader("📊 Price Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    price_change_1d = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
    price_change_7d = ((current_price - data['Close'].iloc[-8]) / data['Close'].iloc[-8]) * 100 if len(data) > 7 else 0
    price_change_30d = ((current_price - data['Close'].iloc[-31]) / data['Close'].iloc[-31]) * 100 if len(data) > 30 else 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.4f}")
        st.metric("24h Change", f"{price_change_1d:.2f}%")
    
    with col2:
        st.metric("7d Change", f"{price_change_7d:.2f}%")
        st.metric("30d Change", f"{price_change_30d:.2f}%")
    
    with col3:
        st.metric("24h High", f"${data['High'].iloc[-1]:.4f}")
        st.metric("24h Low", f"${data['Low'].iloc[-1]:.4f}")
    
    with col4:
        st.metric("24h Volume", f"${data['Volume'].iloc[-1]:,.0f}")
        avg_volume = data['Volume'].mean()
        st.metric("Avg Volume", f"${avg_volume:,.0f}")
    
    # Advanced statistics
    st.subheader("📈 Advanced Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility analysis
        returns = data['Close'].pct_change().dropna()
        volatility_1d = returns.std()
        volatility_annualized = volatility_1d * np.sqrt(365) * 100
        
        st.write("**Volatility Analysis**")
        st.write(f"• Daily Volatility: {volatility_1d*100:.2f}%")
        st.write(f"• Annualized Volatility: {volatility_annualized:.2f}%")
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        st.write(f"• Maximum Drawdown: {max_drawdown:.2f}%")
        
        # VaR calculation (simplified)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        st.write(f"• VaR (95%): {var_95:.2f}%")
        st.write(f"• VaR (99%): {var_99:.2f}%")
    
    with col2:
        # Price levels
        st.write("**Key Price Levels**")
        
        period_high = data['High'].max()
        period_low = data['Low'].min()
        
        # Support and resistance levels
        st.write(f"• Period High: ${period_high:.4f}")
        st.write(f"• Period Low: ${period_low:.4f}")
        
        # Moving averages as support/resistance
        ma_20 = data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) > 50 else None
        
        st.write(f"• MA(20): ${ma_20:.4f}")
        if ma_50:
            st.write(f"• MA(50): ${ma_50:.4f}")
        
        # Current RSI
        rsi = analyzer.calculate_rsi(data['Close'])
        if rsi is not None and not rsi.empty:
            current_rsi = rsi.iloc[-1]
            st.write(f"• Current RSI: {current_rsi:.1f}")
    
    # Export data
    st.subheader("💾 Export Analysis Data")
    
    # Prepare export data
    export_data = data.copy()
    
    # Add technical indicators to export
    if show_ma:
        export_data['MA_20'] = data['Close'].rolling(20).mean()
        export_data['MA_50'] = data['Close'].rolling(50).mean()
    
    if show_rsi:
        rsi = analyzer.calculate_rsi(data['Close'])
        if rsi is not None:
            export_data['RSI'] = rsi
    
    if show_bb:
        bb = analyzer.calculate_bollinger_bands(data['Close'])
        if bb is not None and not bb.empty:
            export_data['BB_Upper'] = bb['Upper']
            export_data['BB_Middle'] = bb['Middle']
            export_data['BB_Lower'] = bb['Lower']
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = export_data.to_csv()
        st.download_button(
            label="📥 Download Analysis CSV",
            data=csv_data,
            file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = export_data.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📥 Download Analysis JSON",
            data=json_data,
            file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
