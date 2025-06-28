import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_fetcher import CryptoDataFetcher
from utils.technical_analysis import TechnicalAnalyzer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Technical Analysis", page_icon="📊", layout="wide")

def main():
    st.title("📊 Comprehensive Technical Analysis")
    st.markdown("Advanced technical indicators and chart pattern analysis")
    
    # Check if data exists in session state
    if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
        st.warning("⚠️ No data available. Please go back to the main page and fetch cryptocurrency data first.")
        return
    
    data = st.session_state.crypto_data
    symbol = st.session_state.selected_crypto
    
    # Initialize technical analyzer
    analyzer = TechnicalAnalyzer()
    
    # Sidebar for indicator selection
    st.sidebar.header("📈 Technical Indicators")
    
    # Trend indicators
    st.sidebar.subheader("Trend Indicators")
    show_ma = st.sidebar.checkbox("Moving Averages", value=True)
    show_ema = st.sidebar.checkbox("Exponential Moving Averages")
    show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_adx = st.sidebar.checkbox("ADX (Trend Strength)")
    
    # Momentum indicators
    st.sidebar.subheader("Momentum Indicators")
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_stoch = st.sidebar.checkbox("Stochastic Oscillator")
    show_williams = st.sidebar.checkbox("Williams %R")
    show_cci = st.sidebar.checkbox("Commodity Channel Index")
    show_momentum = st.sidebar.checkbox("Momentum")
    show_roc = st.sidebar.checkbox("Rate of Change")
    
    # Volatility indicators
    st.sidebar.subheader("Volatility Indicators")
    show_atr = st.sidebar.checkbox("Average True Range")
    
    # Pattern analysis
    st.sidebar.subheader("Pattern Analysis")
    show_support_resistance = st.sidebar.checkbox("Support/Resistance")
    show_fibonacci = st.sidebar.checkbox("Fibonacci Retracements")
    
    # Calculate all indicators
    with st.spinner("Calculating technical indicators..."):
        indicators = analyzer.calculate_all_indicators(data)
    
    # Main analysis display
    if indicators:
        st.success("✅ Technical indicators calculated successfully")
        
        # Trading signals overview
        st.subheader("🎯 Trading Signals Overview")
        
        signals = analyzer.generate_signals(data)
        
        if not signals.empty and 'Composite_Signal' in signals.columns:
            # Current signals
            col1, col2, col3, col4 = st.columns(4)
            
            current_rsi_signal = signals['RSI_Signal'].iloc[-1] if 'RSI_Signal' in signals.columns else 0
            current_macd_signal = signals['MACD_Signal'].iloc[-1] if 'MACD_Signal' in signals.columns else 0
            current_ma_signal = signals['MA_Signal'].iloc[-1] if 'MA_Signal' in signals.columns else 0
            current_composite = signals['Composite_Signal'].iloc[-1]
            
            with col1:
                rsi_label = "🟢 Buy" if current_rsi_signal > 0 else "🔴 Sell" if current_rsi_signal < 0 else "🟡 Neutral"
                st.metric("RSI Signal", rsi_label)
            
            with col2:
                macd_label = "🟢 Buy" if current_macd_signal > 0 else "🔴 Sell" if current_macd_signal < 0 else "🟡 Neutral"
                st.metric("MACD Signal", macd_label)
            
            with col3:
                ma_label = "🟢 Bullish" if current_ma_signal > 0 else "🔴 Bearish"
                st.metric("MA Trend", ma_label)
            
            with col4:
                if current_composite > 0.3:
                    composite_label = "🟢 Strong Buy"
                elif current_composite > 0:
                    composite_label = "🟢 Buy"
                elif current_composite < -0.3:
                    composite_label = "🔴 Strong Sell"
                elif current_composite < 0:
                    composite_label = "🔴 Sell"
                else:
                    composite_label = "🟡 Neutral"
                st.metric("Composite Signal", composite_label)
        
        # Trend Analysis Section
        if show_ma or show_ema or show_bb or show_adx:
            st.subheader("📈 Trend Analysis")
            
            # Moving Averages
            if show_ma and 'Moving_Averages' in indicators:
                st.write("**Moving Averages**")
                ma_data = indicators['Moving_Averages']
                
                fig_ma = go.Figure()
                
                # Price
                fig_ma.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Moving averages
                colors = ['orange', 'red', 'purple', 'brown', 'pink']
                for i, (col, color) in enumerate(zip(ma_data.columns, colors)):
                    if not ma_data[col].isna().all():
                        fig_ma.add_trace(go.Scatter(
                            x=data.index,
                            y=ma_data[col],
                            mode='lines',
                            name=col,
                            line=dict(color=color, width=1)
                        ))
                
                fig_ma.update_layout(
                    title="Moving Averages",
                    yaxis_title="Price (USD)",
                    height=400
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # MA analysis
                current_price = data['Close'].iloc[-1]
                ma_analysis = []
                
                for col in ma_data.columns:
                    if not ma_data[col].isna().all():
                        ma_value = ma_data[col].iloc[-1]
                        if not pd.isna(ma_value):
                            position = "Above" if current_price > ma_value else "Below"
                            distance = abs((current_price - ma_value) / ma_value * 100)
                            ma_analysis.append(f"• {col}: {position} ({distance:.2f}% away)")
                
                if ma_analysis:
                    st.write("**Current Position vs Moving Averages:**")
                    for analysis in ma_analysis:
                        st.write(analysis)
            
            # Exponential Moving Averages
            if show_ema and 'EMA' in indicators:
                st.write("**Exponential Moving Averages**")
                ema_data = indicators['EMA']
                
                fig_ema = go.Figure()
                
                # Price
                fig_ema.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                # EMAs
                colors = ['yellow', 'cyan', 'magenta', 'green']
                for i, (col, color) in enumerate(zip(ema_data.columns, colors)):
                    if not ema_data[col].isna().all():
                        fig_ema.add_trace(go.Scatter(
                            x=data.index,
                            y=ema_data[col],
                            mode='lines',
                            name=col,
                            line=dict(color=color, width=1, dash='dash')
                        ))
                
                fig_ema.update_layout(
                    title="Exponential Moving Averages",
                    yaxis_title="Price (USD)",
                    height=400
                )
                
                st.plotly_chart(fig_ema, use_container_width=True)
            
            # Bollinger Bands
            if show_bb and 'Bollinger_Bands' in indicators:
                st.write("**Bollinger Bands**")
                bb_data = indicators['Bollinger_Bands']
                
                if not bb_data.empty:
                    fig_bb = go.Figure()
                    
                    # Price
                    fig_bb.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Bollinger Bands
                    fig_bb.add_trace(go.Scatter(
                        x=data.index,
                        y=bb_data['Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='red', width=1)
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=data.index,
                        y=bb_data['Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='red', width=1),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.1)'
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=data.index,
                        y=bb_data['Middle'],
                        mode='lines',
                        name='BB Middle (SMA)',
                        line=dict(color='orange', width=1)
                    ))
                    
                    fig_bb.update_layout(
                        title="Bollinger Bands",
                        yaxis_title="Price (USD)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                    # BB Analysis
                    current_price = data['Close'].iloc[-1]
                    bb_upper = bb_data['Upper'].iloc[-1]
                    bb_lower = bb_data['Lower'].iloc[-1]
                    bb_middle = bb_data['Middle'].iloc[-1]
                    
                    bb_position = "Upper Band" if current_price > bb_upper else \
                                 "Lower Band" if current_price < bb_lower else \
                                 "Middle Range"
                    
                    band_width = ((bb_upper - bb_lower) / bb_middle) * 100
                    
                    st.write(f"**Bollinger Band Analysis:**")
                    st.write(f"• Current Position: {bb_position}")
                    st.write(f"• Band Width: {band_width:.2f}% ({'Narrow' if band_width < 10 else 'Wide'})")
            
            # ADX (Average Directional Index)
            if show_adx and 'ADX' in indicators:
                st.write("**ADX - Trend Strength**")
                adx_data = indicators['ADX']
                
                if not adx_data.isna().all():
                    fig_adx = go.Figure()
                    
                    fig_adx.add_trace(go.Scatter(
                        x=data.index,
                        y=adx_data,
                        mode='lines',
                        name='ADX',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # ADX levels
                    fig_adx.add_hline(y=25, line_dash="dash", line_color="green", 
                                     annotation_text="Strong Trend (25)")
                    fig_adx.add_hline(y=50, line_dash="dash", line_color="red", 
                                     annotation_text="Very Strong Trend (50)")
                    
                    fig_adx.update_layout(
                        title="ADX - Average Directional Index",
                        yaxis_title="ADX Value",
                        height=300
                    )
                    
                    st.plotly_chart(fig_adx, use_container_width=True)
                    
                    current_adx = adx_data.iloc[-1]
                    if not pd.isna(current_adx):
                        if current_adx > 50:
                            trend_strength = "Very Strong Trend"
                        elif current_adx > 25:
                            trend_strength = "Strong Trend"
                        else:
                            trend_strength = "Weak/No Trend"
                        
                        st.write(f"• Current ADX: {current_adx:.1f} ({trend_strength})")
        
        # Momentum Analysis Section
        momentum_indicators = [show_rsi, show_macd, show_stoch, show_williams, show_cci, show_momentum, show_roc]
        if any(momentum_indicators):
            st.subheader("⚡ Momentum Analysis")
            
            # RSI
            if show_rsi and 'RSI' in indicators:
                st.write("**RSI - Relative Strength Index**")
                rsi_data = indicators['RSI']
                
                if not rsi_data.isna().all():
                    fig_rsi = go.Figure()
                    
                    fig_rsi.add_trace(go.Scatter(
                        x=data.index,
                        y=rsi_data,
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # RSI levels
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                     annotation_text="Oversold (30)")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                     annotation_text="Midline")
                    
                    fig_rsi.update_layout(
                        title="RSI - Relative Strength Index",
                        yaxis_title="RSI",
                        yaxis=dict(range=[0, 100]),
                        height=300
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    current_rsi = rsi_data.iloc[-1]
                    if not pd.isna(current_rsi):
                        if current_rsi > 70:
                            rsi_status = "🔴 Overbought"
                        elif current_rsi < 30:
                            rsi_status = "🟢 Oversold"
                        else:
                            rsi_status = "🟡 Neutral"
                        
                        st.write(f"• Current RSI: {current_rsi:.1f} ({rsi_status})")
            
            # MACD
            if show_macd and 'MACD' in indicators:
                st.write("**MACD - Moving Average Convergence Divergence**")
                macd_data = indicators['MACD']
                
                if not macd_data.empty:
                    fig_macd = go.Figure()
                    
                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=macd_data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=macd_data['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_macd.add_trace(go.Bar(
                        x=data.index,
                        y=macd_data['Histogram'],
                        name='Histogram',
                        marker_color='gray',
                        opacity=0.6
                    ))
                    
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    fig_macd.update_layout(
                        title="MACD - Moving Average Convergence Divergence",
                        yaxis_title="MACD",
                        height=350
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    current_macd = macd_data['MACD'].iloc[-1]
                    current_signal = macd_data['Signal'].iloc[-1]
                    
                    if not pd.isna(current_macd) and not pd.isna(current_signal):
                        macd_signal = "🟢 Bullish" if current_macd > current_signal else "🔴 Bearish"
                        st.write(f"• MACD vs Signal: {macd_signal}")
                        st.write(f"• MACD: {current_macd:.4f}, Signal: {current_signal:.4f}")
            
            # Stochastic Oscillator
            if show_stoch and 'Stochastic' in indicators:
                st.write("**Stochastic Oscillator**")
                stoch_data = indicators['Stochastic']
                
                if not stoch_data.empty:
                    fig_stoch = go.Figure()
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=data.index,
                        y=stoch_data['%K'],
                        mode='lines',
                        name='%K',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=data.index,
                        y=stoch_data['%D'],
                        mode='lines',
                        name='%D',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Stochastic levels
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
            
            # Williams %R
            if show_williams and 'Williams_R' in indicators:
                st.write("**Williams %R**")
                williams_data = indicators['Williams_R']
                
                if not williams_data.isna().all():
                    fig_williams = go.Figure()
                    
                    fig_williams.add_trace(go.Scatter(
                        x=data.index,
                        y=williams_data,
                        mode='lines',
                        name='Williams %R',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Williams %R levels
                    fig_williams.add_hline(y=-20, line_dash="dash", line_color="red", 
                                          annotation_text="Overbought (-20)")
                    fig_williams.add_hline(y=-80, line_dash="dash", line_color="green", 
                                          annotation_text="Oversold (-80)")
                    
                    fig_williams.update_layout(
                        title="Williams %R",
                        yaxis_title="Williams %R",
                        yaxis=dict(range=[-100, 0]),
                        height=300
                    )
                    
                    st.plotly_chart(fig_williams, use_container_width=True)
            
            # Other momentum indicators (CCI, Momentum, ROC)
            other_momentum = []
            if show_cci and 'CCI' in indicators:
                other_momentum.append(('CCI', indicators['CCI'], 'orange'))
            if show_momentum and 'Momentum' in indicators:
                other_momentum.append(('Momentum', indicators['Momentum'], 'purple'))
            if show_roc and 'ROC' in indicators:
                other_momentum.append(('ROC', indicators['ROC'], 'brown'))
            
            if other_momentum:
                for name, data_series, color in other_momentum:
                    if not data_series.isna().all():
                        fig_other = go.Figure()
                        
                        fig_other.add_trace(go.Scatter(
                            x=data.index,
                            y=data_series,
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=2)
                        ))
                        
                        if name == 'CCI':
                            fig_other.add_hline(y=100, line_dash="dash", line_color="red")
                            fig_other.add_hline(y=-100, line_dash="dash", line_color="green")
                        
                        fig_other.add_hline(y=0, line_dash="dot", line_color="gray")
                        
                        fig_other.update_layout(
                            title=f"{name} - {name}",
                            yaxis_title=name,
                            height=300
                        )
                        
                        st.plotly_chart(fig_other, use_container_width=True)
        
        # Volatility Analysis
        if show_atr and 'ATR' in indicators:
            st.subheader("📊 Volatility Analysis")
            
            st.write("**ATR - Average True Range**")
            atr_data = indicators['ATR']
            
            if not atr_data.isna().all():
                fig_atr = go.Figure()
                
                fig_atr.add_trace(go.Scatter(
                    x=data.index,
                    y=atr_data,
                    mode='lines',
                    name='ATR',
                    line=dict(color='red', width=2)
                ))
                
                fig_atr.update_layout(
                    title="ATR - Average True Range",
                    yaxis_title="ATR Value",
                    height=300
                )
                
                st.plotly_chart(fig_atr, use_container_width=True)
                
                current_atr = atr_data.iloc[-1]
                avg_atr = atr_data.mean()
                
                if not pd.isna(current_atr) and not pd.isna(avg_atr):
                    volatility_status = "High" if current_atr > avg_atr * 1.2 else \
                                      "Low" if current_atr < avg_atr * 0.8 else "Normal"
                    st.write(f"• Current ATR: {current_atr:.4f}")
                    st.write(f"• Average ATR: {avg_atr:.4f}")
                    st.write(f"• Volatility Status: {volatility_status}")
        
        # Pattern Analysis
        if show_support_resistance or show_fibonacci:
            st.subheader("🎯 Pattern Analysis")
            
            # Support and Resistance
            if show_support_resistance:
                st.write("**Support and Resistance Levels**")
                
                sr_analysis = analyzer.identify_support_resistance(data['Close'])
                
                if sr_analysis:
                    fig_sr = go.Figure()
                    
                    # Price chart
                    fig_sr.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol
                    ))
                    
                    # Support levels
                    if not sr_analysis['support_levels'].empty:
                        for idx, support_price in sr_analysis['support_levels'].items():
                            fig_sr.add_hline(
                                y=support_price,
                                line_dash="dot",
                                line_color="green",
                                annotation_text=f"Support: ${support_price:.2f}"
                            )
                    
                    # Resistance levels
                    if not sr_analysis['resistance_levels'].empty:
                        for idx, resistance_price in sr_analysis['resistance_levels'].items():
                            fig_sr.add_hline(
                                y=resistance_price,
                                line_dash="dot",
                                line_color="red",
                                annotation_text=f"Resistance: ${resistance_price:.2f}"
                            )
                    
                    fig_sr.update_layout(
                        title="Support and Resistance Levels",
                        yaxis_title="Price (USD)",
                        height=500,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig_sr, use_container_width=True)
            
            # Fibonacci Retracements
            if show_fibonacci:
                st.write("**Fibonacci Retracement Levels**")
                
                period_high = data['High'].max()
                period_low = data['Low'].min()
                
                fib_levels = analyzer.calculate_fibonacci_retracement(period_high, period_low)
                
                if fib_levels:
                    fig_fib = go.Figure()
                    
                    # Price chart
                    fig_fib.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol
                    ))
                    
                    # Fibonacci levels
                    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
                    for i, (level, price) in enumerate(fib_levels.items()):
                        fig_fib.add_hline(
                            y=price,
                            line_dash="dash",
                            line_color=colors[i % len(colors)],
                            annotation_text=f"Fib {level}: ${price:.2f}"
                        )
                    
                    fig_fib.update_layout(
                        title="Fibonacci Retracement Levels",
                        yaxis_title="Price (USD)",
                        height=500,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig_fib, use_container_width=True)
                    
                    # Current price vs Fibonacci levels
                    current_price = data['Close'].iloc[-1]
                    st.write(f"**Current Price Analysis (${current_price:.4f}):**")
                    
                    for level, price in fib_levels.items():
                        distance = abs((current_price - price) / price * 100)
                        position = "Above" if current_price > price else "Below"
                        st.write(f"• {level}: {position} by {distance:.2f}%")
        
        # Technical Summary
        st.subheader("📋 Technical Analysis Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Current Market Condition**")
            
            # Determine overall trend
            if 'Moving_Averages' in indicators and 'MA_20' in indicators['Moving_Averages'].columns:
                ma_20 = indicators['Moving_Averages']['MA_20'].iloc[-1]
                ma_50 = indicators['Moving_Averages']['MA_50'].iloc[-1] if 'MA_50' in indicators['Moving_Averages'].columns else None
                current_price = data['Close'].iloc[-1]
                
                if ma_50 and not pd.isna(ma_20) and not pd.isna(ma_50):
                    if current_price > ma_20 > ma_50:
                        trend = "🟢 Strong Uptrend"
                    elif current_price > ma_50 > ma_20:
                        trend = "🔴 Strong Downtrend"
                    elif current_price > ma_20:
                        trend = "🟡 Short-term Bullish"
                    else:
                        trend = "🟡 Short-term Bearish"
                else:
                    trend = "📊 Trend Analysis Incomplete"
                
                st.write(f"• Overall Trend: {trend}")
            
            # Momentum summary
            if 'RSI' in indicators:
                current_rsi = indicators['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    if current_rsi > 70:
                        momentum = "🔴 Overbought"
                    elif current_rsi < 30:
                        momentum = "🟢 Oversold"
                    else:
                        momentum = "🟡 Neutral"
                    st.write(f"• Momentum: {momentum}")
            
            # Volatility summary
            if 'ATR' in indicators:
                current_atr = indicators['ATR'].iloc[-1]
                avg_atr = indicators['ATR'].rolling(20).mean().iloc[-1]
                
                if not pd.isna(current_atr) and not pd.isna(avg_atr):
                    if current_atr > avg_atr * 1.2:
                        volatility = "🔴 High Volatility"
                    elif current_atr < avg_atr * 0.8:
                        volatility = "🟢 Low Volatility"
                    else:
                        volatility = "🟡 Normal Volatility"
                    st.write(f"• Volatility: {volatility}")
        
        with summary_col2:
            st.write("**Trading Recommendations**")
            
            if not signals.empty:
                composite_signal = signals['Composite_Signal'].iloc[-1]
                
                if composite_signal > 0.5:
                    recommendation = "🟢 STRONG BUY"
                    confidence = "High"
                elif composite_signal > 0.2:
                    recommendation = "🟢 BUY"
                    confidence = "Medium"
                elif composite_signal < -0.5:
                    recommendation = "🔴 STRONG SELL"
                    confidence = "High"
                elif composite_signal < -0.2:
                    recommendation = "🔴 SELL"
                    confidence = "Medium"
                else:
                    recommendation = "🟡 HOLD"
                    confidence = "Low"
                
                st.write(f"• Signal: {recommendation}")
                st.write(f"• Confidence: {confidence}")
                st.write(f"• Composite Score: {composite_signal:.3f}")
            
            # Risk level
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365) * 100
            
            if volatility > 100:
                risk_level = "🔴 Very High Risk"
            elif volatility > 50:
                risk_level = "🟡 High Risk"
            elif volatility > 25:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🟢 Low Risk"
            
            st.write(f"• Risk Level: {risk_level}")
            st.write(f"• Annualized Volatility: {volatility:.1f}%")
        
        # Export technical analysis
        st.subheader("💾 Export Technical Analysis")
        
        # Combine all indicators into export data
        export_data = data.copy()
        
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, pd.DataFrame):
                for col in indicator_data.columns:
                    export_data[f"{indicator_name}_{col}"] = indicator_data[col]
            elif isinstance(indicator_data, pd.Series):
                export_data[indicator_name] = indicator_data
        
        # Add signals
        if not signals.empty:
            for col in signals.columns:
                export_data[f"Signal_{col}"] = signals[col]
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_data.to_csv()
            st.download_button(
                label="📥 Download Technical Analysis CSV",
                data=csv_data,
                file_name=f"{symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = export_data.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download Technical Analysis JSON",
                data=json_data,
                file_name=f"{symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        st.error("❌ Failed to calculate technical indicators")

if __name__ == "__main__":
    main()
