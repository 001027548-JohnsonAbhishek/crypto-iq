import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.trading_signals import TradingSignalGenerator
from utils.technical_analysis import TechnicalAnalyzer
from utils.data_fetcher import CryptoDataFetcher
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trading Signals", page_icon="⚡", layout="wide")

def main():
    st.title("⚡ Advanced Trading Signals & Automation")
    st.markdown("Comprehensive trading signal analysis with backtesting and automated alerts")
    
    # Check if data exists in session state
    if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
        st.warning("⚠️ No data available. Please go back to the main page and fetch cryptocurrency data first.")
        return
    
    data = st.session_state.crypto_data
    symbol = st.session_state.selected_crypto
    crypto_name = symbol.split('-')[0]
    
    # Initialize signal generator
    signal_generator = TradingSignalGenerator()
    
    # Sidebar for signal configuration
    st.sidebar.header("🔧 Signal Configuration")
    
    # Signal types selection
    st.sidebar.subheader("Signal Types")
    
    enable_ma_signals = st.sidebar.checkbox("Moving Average Signals", value=True)
    enable_rsi_signals = st.sidebar.checkbox("RSI Signals", value=True)
    enable_macd_signals = st.sidebar.checkbox("MACD Signals", value=True)
    enable_bb_signals = st.sidebar.checkbox("Bollinger Bands Signals", value=True)
    enable_stoch_signals = st.sidebar.checkbox("Stochastic Signals")
    enable_volume_signals = st.sidebar.checkbox("Volume Signals")
    enable_sr_signals = st.sidebar.checkbox("Support/Resistance Signals")
    enable_trend_signals = st.sidebar.checkbox("Trend Following Signals")
    enable_momentum_signals = st.sidebar.checkbox("Momentum Signals")
    
    # Signal sensitivity
    st.sidebar.subheader("Signal Sensitivity")
    
    signal_threshold = st.sidebar.slider(
        "Signal Threshold",
        0.1, 1.0, 0.3, 0.1,
        help="Minimum signal strength to trigger recommendations"
    )
    
    # Alert settings
    st.sidebar.subheader("Alert Settings")
    
    enable_price_alerts = st.sidebar.checkbox("Price Threshold Alerts")
    
    if enable_price_alerts:
        current_price = data['Close'].iloc[-1]
        
        upper_threshold = st.sidebar.number_input(
            "Upper Price Alert ($)",
            min_value=0.0,
            value=float(current_price * 1.1),
            step=0.01,
            format="%.4f"
        )
        
        lower_threshold = st.sidebar.number_input(
            "Lower Price Alert ($)",
            min_value=0.0,
            value=float(current_price * 0.9),
            step=0.01,
            format="%.4f"
        )
        
        price_thresholds = {
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold
        }
    else:
        price_thresholds = None
    
    # Backtesting parameters
    st.sidebar.subheader("Backtesting Settings")
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        value=10000,
        step=1000
    )
    
    # Generate signals
    if st.sidebar.button("🚀 Generate Trading Signals", type="primary"):
        with st.spinner("Generating comprehensive trading signals..."):
            # Generate signals
            signals = signal_generator.generate_comprehensive_signals(data)
            
            if signals is not None and not signals.empty:
                st.session_state.trading_signals = signals
                st.session_state.signal_params = {
                    'threshold': signal_threshold,
                    'price_thresholds': price_thresholds,
                    'initial_capital': initial_capital
                }
                st.success("✅ Trading signals generated successfully!")
            else:
                st.error("❌ Failed to generate trading signals")
    
    # Display signals if available
    if 'trading_signals' in st.session_state:
        signals = st.session_state.trading_signals
        params = st.session_state.signal_params
        
        # Current Signal Dashboard
        st.subheader("🎯 Current Signal Dashboard")
        
        # Get latest signals
        latest_signals = signals.tail(1)
        current_composite = latest_signals['Composite_Signal'].iloc[0]
        current_recommendation = latest_signals['Recommendation'].iloc[0]
        current_strength = latest_signals['Signal_Strength'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Recommendation with color coding
            rec_color = "🟢" if "BUY" in current_recommendation else "🔴" if "SELL" in current_recommendation else "🟡"
            st.metric("Current Signal", f"{rec_color} {current_recommendation}")
        
        with col2:
            st.metric("Signal Strength", f"{current_strength:.2f}")
        
        with col3:
            composite_direction = "Bullish" if current_composite > 0 else "Bearish" if current_composite < 0 else "Neutral"
            st.metric("Market Bias", composite_direction)
        
        with col4:
            # Count recent signals
            recent_signals = signals.tail(10)
            buy_signals = len(recent_signals[recent_signals['Recommendation'].str.contains('BUY', na=False)])
            sell_signals = len(recent_signals[recent_signals['Recommendation'].str.contains('SELL', na=False)])
            signal_trend = "Bullish Trend" if buy_signals > sell_signals else "Bearish Trend" if sell_signals > buy_signals else "Mixed Signals"
            st.metric("Recent Trend", signal_trend)
        
        # Individual Signal Analysis
        st.subheader("📊 Individual Signal Analysis")
        
        # Create tabs for different signal types
        tab1, tab2, tab3, tab4 = st.tabs(["Technical Signals", "Momentum Signals", "Volume Analysis", "Trend Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Signal
                if 'RSI_Signal' in signals.columns:
                    current_rsi_signal = signals['RSI_Signal'].iloc[-1]
                    rsi_status = "🟢 Buy" if current_rsi_signal > 0 else "🔴 Sell" if current_rsi_signal < 0 else "🟡 Neutral"
                    st.metric("RSI Signal", rsi_status)
                
                # MACD Signal
                if 'MACD_Signal' in signals.columns:
                    current_macd_signal = signals['MACD_Signal'].iloc[-1]
                    macd_status = "🟢 Bullish" if current_macd_signal > 0 else "🔴 Bearish" if current_macd_signal < 0 else "🟡 Neutral"
                    st.metric("MACD Signal", macd_status)
            
            with col2:
                # Bollinger Bands Signal
                if 'BB_Signal' in signals.columns:
                    current_bb_signal = signals['BB_Signal'].iloc[-1]
                    bb_status = "🟢 Oversold" if current_bb_signal > 0 else "🔴 Overbought" if current_bb_signal < 0 else "🟡 Neutral"
                    st.metric("Bollinger Bands", bb_status)
                
                # Moving Average Signal
                if 'MA_Signal' in signals.columns:
                    current_ma_signal = signals['MA_Signal'].iloc[-1]
                    ma_status = "🟢 Bullish" if current_ma_signal > 0 else "🔴 Bearish"
                    st.metric("MA Trend", ma_status)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Stochastic Signal
                if 'Stoch_Signal' in signals.columns:
                    current_stoch_signal = signals['Stoch_Signal'].iloc[-1]
                    stoch_status = "🟢 Buy" if current_stoch_signal > 0 else "🔴 Sell" if current_stoch_signal < 0 else "🟡 Neutral"
                    st.metric("Stochastic Signal", stoch_status)
                
                # Momentum Signal
                if 'Momentum_Signal' in signals.columns:
                    current_momentum_signal = signals['Momentum_Signal'].iloc[-1]
                    momentum_status = "🟢 Accelerating" if current_momentum_signal > 0 else "🔴 Decelerating" if current_momentum_signal < 0 else "🟡 Stable"
                    st.metric("Momentum", momentum_status)
            
            with col2:
                # Additional momentum metrics
                analyzer = TechnicalAnalyzer()
                rsi = analyzer.calculate_rsi(data['Close'])
                if not rsi.empty:
                    current_rsi = rsi.iloc[-1]
                    st.metric("Current RSI", f"{current_rsi:.1f}")
                
                # ROC if available
                roc = analyzer.calculate_roc(data['Close'])
                if not roc.empty:
                    current_roc = roc.iloc[-1]
                    st.metric("Rate of Change", f"{current_roc:.2f}%")
        
        with tab3:
            # Volume analysis
            if 'Volume_Signal' in signals.columns:
                current_volume_signal = signals['Volume_Signal'].iloc[-1]
                volume_status = "🟢 Confirming" if abs(current_volume_signal) > 0.3 else "🟡 Weak"
                st.metric("Volume Confirmation", volume_status)
            
            # Volume metrics
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Volume", f"${current_volume:,.0f}")
            
            with col2:
                st.metric("20-day Avg Volume", f"${avg_volume:,.0f}")
            
            with col3:
                volume_color = "🟢" if volume_ratio > 1.5 else "🔴" if volume_ratio < 0.5 else "🟡"
                st.metric("Volume Ratio", f"{volume_color} {volume_ratio:.2f}x")
        
        with tab4:
            # Trend analysis
            if 'Trend_Signal' in signals.columns:
                current_trend_signal = signals['Trend_Signal'].iloc[-1]
                trend_status = "🟢 Strong Uptrend" if current_trend_signal > 0.3 else "🔴 Strong Downtrend" if current_trend_signal < -0.3 else "🟡 Sideways"
                st.metric("Trend Direction", trend_status)
            
            # Support/Resistance
            if 'SR_Signal' in signals.columns:
                current_sr_signal = signals['SR_Signal'].iloc[-1]
                sr_status = "🟢 Breakout" if current_sr_signal > 0 else "🔴 Breakdown" if current_sr_signal < 0 else "🟡 Range-bound"
                st.metric("S/R Status", sr_status)
        
        # Signal History Chart
        st.subheader("📈 Signal History & Price Action")
        
        # Create comprehensive chart
        fig = go.Figure()
        
        # Price chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=crypto_name,
            yaxis="y1"
        ))
        
        # Add buy/sell signals as markers
        buy_signals_mask = signals['Recommendation'].str.contains('BUY', na=False)
        sell_signals_mask = signals['Recommendation'].str.contains('SELL', na=False)
        
        if buy_signals_mask.any():
            buy_dates = signals[buy_signals_mask].index
            buy_prices = data.loc[buy_dates, 'Close']
            
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Signals',
                yaxis="y1"
            ))
        
        if sell_signals_mask.any():
            sell_dates = signals[sell_signals_mask].index
            sell_prices = data.loc[sell_dates, 'Close']
            
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Signals',
                yaxis="y1"
            ))
        
        # Composite signal as secondary y-axis
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['Composite_Signal'],
            mode='lines',
            name='Composite Signal',
            line=dict(color='purple', width=2),
            yaxis="y2"
        ))
        
        # Add signal threshold lines
        fig.add_hline(y=signal_threshold, line_dash="dash", line_color="green", 
                     annotation_text=f"Buy Threshold ({signal_threshold})")
        fig.add_hline(y=-signal_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Sell Threshold ({-signal_threshold})")
        
        # Update layout for dual y-axis
        fig.update_layout(
            title=f"{crypto_name} Price Action with Trading Signals",
            xaxis_title="Date",
            yaxis=dict(title="Price (USD)", side="left"),
            yaxis2=dict(title="Signal Strength", side="right", overlaying="y", range=[-1, 1]),
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal Statistics
        st.subheader("📊 Signal Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Signal Distribution**")
            
            buy_count = len(signals[signals['Recommendation'].str.contains('BUY', na=False)])
            sell_count = len(signals[signals['Recommendation'].str.contains('SELL', na=False)])
            hold_count = len(signals[signals['Recommendation'] == 'HOLD'])
            
            st.write(f"• Buy Signals: {buy_count}")
            st.write(f"• Sell Signals: {sell_count}")
            st.write(f"• Hold Signals: {hold_count}")
            
            if buy_count + sell_count > 0:
                signal_ratio = buy_count / (buy_count + sell_count)
                st.write(f"• Buy/Sell Ratio: {signal_ratio:.2f}")
        
        with col2:
            st.write("**Signal Quality**")
            
            # Signal strength statistics
            strong_signals = len(signals[signals['Signal_Strength'] > 0.7])
            moderate_signals = len(signals[(signals['Signal_Strength'] > 0.3) & (signals['Signal_Strength'] <= 0.7)])
            weak_signals = len(signals[signals['Signal_Strength'] <= 0.3])
            
            st.write(f"• Strong Signals (>0.7): {strong_signals}")
            st.write(f"• Moderate Signals (0.3-0.7): {moderate_signals}")
            st.write(f"• Weak Signals (≤0.3): {weak_signals}")
            
            avg_strength = signals['Signal_Strength'].mean()
            st.write(f"• Average Strength: {avg_strength:.3f}")
        
        with col3:
            st.write("**Recent Performance**")
            
            # Last 30 days performance
            recent_signals = signals.tail(30)
            recent_composite = recent_signals['Composite_Signal'].mean()
            
            if recent_composite > 0.1:
                recent_bias = "🟢 Bullish Bias"
            elif recent_composite < -0.1:
                recent_bias = "🔴 Bearish Bias"
            else:
                recent_bias = "🟡 Neutral Bias"
            
            st.write(f"• 30-day Bias: {recent_bias}")
            st.write(f"• Average Signal: {recent_composite:.3f}")
            
            # Signal consistency
            signal_volatility = recent_signals['Composite_Signal'].std()
            consistency = "High" if signal_volatility < 0.3 else "Medium" if signal_volatility < 0.6 else "Low"
            st.write(f"• Signal Consistency: {consistency}")
        
        # Backtesting Results
        st.subheader("📈 Backtesting Results")
        
        if st.button("🔍 Run Backtest"):
            with st.spinner("Running signal backtesting..."):
                backtest_results = signal_generator.backtest_signals(
                    data, signals, initial_capital=params['initial_capital']
                )
            
            if backtest_results:
                st.session_state.backtest_results = backtest_results
        
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{results['total_return']*100:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Buy & Hold Return",
                    f"{results['buy_hold_return']*100:+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Excess Return",
                    f"{results['excess_return']*100:+.2f}%"
                )
            
            with col4:
                st.metric(
                    "Number of Trades",
                    f"{results['num_trades']}"
                )
            
            # Portfolio value chart
            if results['portfolio_history']:
                portfolio_df = pd.DataFrame(results['portfolio_history'])
                
                fig_backtest = go.Figure()
                
                # Portfolio value
                fig_backtest.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Signal Strategy',
                    line=dict(color='blue', width=2)
                ))
                
                # Buy and hold comparison
                buy_hold_values = []
                initial_price = data['Close'].iloc[0]
                shares = params['initial_capital'] / initial_price
                
                for _, row in portfolio_df.iterrows():
                    current_price = row['price']
                    buy_hold_value = shares * current_price
                    buy_hold_values.append(buy_hold_value)
                
                fig_backtest.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=buy_hold_values,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig_backtest.update_layout(
                    title="Backtesting Results: Signal Strategy vs Buy & Hold",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                
                st.plotly_chart(fig_backtest, use_container_width=True)
            
            # Trade history
            if results['trades']:
                st.write("**Trade History**")
                
                trades_df = pd.DataFrame(results['trades'])
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df = trades_df.sort_values('date', ascending=False)
                
                # Show recent trades
                st.dataframe(
                    trades_df.head(10)[['date', 'action', 'price', 'shares', 'portfolio_value']],
                    use_container_width=True
                )
                
                # Trade analysis
                if len(trades_df) > 1:
                    buy_trades = trades_df[trades_df['action'] == 'BUY']
                    sell_trades = trades_df[trades_df['action'] == 'SELL']
                    
                    if len(buy_trades) > 0 and len(sell_trades) > 0:
                        avg_buy_price = buy_trades['price'].mean()
                        avg_sell_price = sell_trades['price'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Average Buy Price", f"${avg_buy_price:.4f}")
                        
                        with col2:
                            st.metric("Average Sell Price", f"${avg_sell_price:.4f}")
                        
                        with col3:
                            if avg_buy_price > 0:
                                avg_trade_return = ((avg_sell_price - avg_buy_price) / avg_buy_price) * 100
                                st.metric("Avg Trade Return", f"{avg_trade_return:+.2f}%")
        
        # Risk-Reward Analysis
        st.subheader("⚖️ Risk-Reward Analysis")
        
        if st.button("📊 Calculate Risk-Reward"):
            with st.spinner("Calculating risk-reward ratios..."):
                risk_reward_ratios = signal_generator.calculate_risk_reward_ratio(
                    signals, data, lookforward_days=30
                )
            
            if risk_reward_ratios:
                st.session_state.risk_reward_ratios = risk_reward_ratios
        
        if 'risk_reward_ratios' in st.session_state:
            rr_ratios = st.session_state.risk_reward_ratios
            
            if rr_ratios:
                rr_df = pd.DataFrame(rr_ratios)
                
                # Risk-reward statistics
                avg_rr_ratio = rr_df['risk_reward_ratio'].mean()
                good_ratios = len(rr_df[rr_df['risk_reward_ratio'] > 2])
                total_signals = len(rr_df)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average R/R Ratio", f"{avg_rr_ratio:.2f}")
                
                with col2:
                    st.metric("Good Ratios (>2:1)", f"{good_ratios}/{total_signals}")
                
                with col3:
                    success_rate = (good_ratios / total_signals) * 100 if total_signals > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Risk-reward distribution
                fig_rr = go.Figure()
                
                fig_rr.add_trace(go.Histogram(
                    x=rr_df['risk_reward_ratio'],
                    nbinsx=20,
                    name="Risk-Reward Distribution"
                ))
                
                fig_rr.add_vline(x=2, line_dash="dash", line_color="green", 
                               annotation_text="Good R/R (2:1)")
                fig_rr.add_vline(x=1, line_dash="dash", line_color="orange", 
                               annotation_text="Breakeven (1:1)")
                
                fig_rr.update_layout(
                    title="Risk-Reward Ratio Distribution",
                    xaxis_title="Risk-Reward Ratio",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig_rr, use_container_width=True)
                
                # Recent risk-reward analysis
                st.write("**Recent Signal Analysis**")
                recent_rr = rr_df.tail(5)
                st.dataframe(
                    recent_rr[['date', 'entry_price', 'potential_reward', 'potential_risk', 'risk_reward_ratio']].round(4),
                    use_container_width=True
                )
        
        # Alert System
        st.subheader("🚨 Active Alerts")
        
        # Generate alerts
        alerts = signal_generator.generate_alerts(data, signals, params['price_thresholds'])
        
        if alerts:
            st.warning(f"⚠️ {len(alerts)} active alert(s) detected!")
            
            for alert in alerts:
                alert_color = "🔴" if alert['type'] in ['SELL_SIGNAL', 'PRICE_ALERT'] else "🟢" if alert['type'] == 'BUY_SIGNAL' else "🟡"
                
                with st.expander(f"{alert_color} {alert['type']} - {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Message:** {alert['message']}")
                    st.write(f"**Current Price:** ${alert['price']:.4f}")
                    
                    if 'strength' in alert:
                        st.write(f"**Signal Strength:** {alert['strength']:.3f}")
        else:
            st.success("✅ No active alerts at this time")
        
        # Signal Configuration Summary
        st.subheader("⚙️ Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Active Signal Types**")
            active_signals = []
            if enable_ma_signals:
                active_signals.append("Moving Average")
            if enable_rsi_signals:
                active_signals.append("RSI")
            if enable_macd_signals:
                active_signals.append("MACD")
            if enable_bb_signals:
                active_signals.append("Bollinger Bands")
            if enable_stoch_signals:
                active_signals.append("Stochastic")
            if enable_volume_signals:
                active_signals.append("Volume")
            if enable_sr_signals:
                active_signals.append("Support/Resistance")
            if enable_trend_signals:
                active_signals.append("Trend Following")
            if enable_momentum_signals:
                active_signals.append("Momentum")
            
            for signal_type in active_signals:
                st.write(f"• {signal_type}")
        
        with col2:
            st.write("**Parameters**")
            st.write(f"• Signal Threshold: {signal_threshold}")
            st.write(f"• Initial Capital: ${params['initial_capital']:,}")
            
            if params['price_thresholds']:
                st.write(f"• Upper Alert: ${params['price_thresholds']['upper_threshold']:.4f}")
                st.write(f"• Lower Alert: ${params['price_thresholds']['lower_threshold']:.4f}")
        
        # Export signals
        st.subheader("💾 Export Trading Signals")
        
        # Prepare export data
        export_signals = signals.copy()
        export_signals['Date'] = export_signals.index
        export_signals['Symbol'] = symbol
        export_signals['Current_Price'] = data['Close']
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_signals.to_csv(index=False)
            st.download_button(
                label="📥 Download Signals CSV",
                data=csv_data,
                file_name=f"{crypto_name}_trading_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = export_signals.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download Signals JSON",
                data=json_data,
                file_name=f"{crypto_name}_trading_signals_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        # Trading signals guide
        st.info("👆 Configure your signal preferences and click 'Generate Trading Signals' to start")
        
        st.subheader("📚 Trading Signals Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Signal Types**")
            st.write("• **Moving Average**: Trend following signals based on MA crossovers")
            st.write("• **RSI**: Momentum signals from overbought/oversold conditions")
            st.write("• **MACD**: Trend and momentum convergence/divergence signals")
            st.write("• **Bollinger Bands**: Volatility-based mean reversion signals")
            st.write("• **Stochastic**: Momentum oscillator for entry/exit timing")
            st.write("• **Volume**: Confirmation signals based on trading activity")
            st.write("• **Support/Resistance**: Breakout and breakdown signals")
            
            st.write("**Risk Management**")
            st.write("• Set appropriate position sizes")
            st.write("• Use stop-loss orders")
            st.write("• Diversify across multiple signals")
            st.write("• Monitor risk-reward ratios")
        
        with col2:
            st.write("**Signal Interpretation**")
            st.write("• **Strong Buy/Sell**: High confidence signals (>0.6 strength)")
            st.write("• **Buy/Sell**: Moderate confidence signals (0.3-0.6 strength)")
            st.write("• **Hold**: Weak or conflicting signals (<0.3 strength)")
            
            st.write("**Best Practices**")
            st.write("• Combine multiple signal types for confirmation")
            st.write("• Consider market context and news events")
            st.write("• Backtest strategies before live trading")
            st.write("• Use alerts for timely signal notifications")
            st.write("• Review and adjust signal parameters regularly")
            
            st.write("**Backtesting Benefits**")
            st.write("• Evaluate strategy performance")
            st.write("• Compare against buy-and-hold")
            st.write("• Identify optimal parameters")
            st.write("• Assess risk-adjusted returns")

if __name__ == "__main__":
    main()
