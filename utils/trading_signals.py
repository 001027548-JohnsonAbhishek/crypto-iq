import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.technical_analysis import TechnicalAnalyzer
import warnings
warnings.filterwarnings('ignore')

class TradingSignalGenerator:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.signals_history = []
        
    def generate_comprehensive_signals(self, data):
        """Generate comprehensive trading signals"""
        try:
            signals = pd.DataFrame(index=data.index)
            
            # Technical indicators
            indicators = self.technical_analyzer.calculate_all_indicators(data)
            
            # Moving Average Signals
            ma_signals = self._generate_ma_signals(data, indicators)
            signals['MA_Signal'] = ma_signals
            
            # RSI Signals
            rsi_signals = self._generate_rsi_signals(indicators.get('RSI'))
            signals['RSI_Signal'] = rsi_signals
            
            # MACD Signals
            macd_signals = self._generate_macd_signals(indicators.get('MACD'))
            signals['MACD_Signal'] = macd_signals
            
            # Bollinger Bands Signals
            bb_signals = self._generate_bollinger_signals(data, indicators.get('Bollinger_Bands'))
            signals['BB_Signal'] = bb_signals
            
            # Stochastic Signals
            stoch_signals = self._generate_stochastic_signals(indicators.get('Stochastic'))
            signals['Stoch_Signal'] = stoch_signals
            
            # Volume Analysis Signals
            volume_signals = self._generate_volume_signals(data)
            signals['Volume_Signal'] = volume_signals
            
            # Support/Resistance Signals
            sr_signals = self._generate_support_resistance_signals(data)
            signals['SR_Signal'] = sr_signals
            
            # Trend Following Signals
            trend_signals = self._generate_trend_signals(data, indicators)
            signals['Trend_Signal'] = trend_signals
            
            # Momentum Signals
            momentum_signals = self._generate_momentum_signals(indicators)
            signals['Momentum_Signal'] = momentum_signals
            
            # Composite Signal
            signal_columns = ['MA_Signal', 'RSI_Signal', 'MACD_Signal', 'BB_Signal', 
                            'Stoch_Signal', 'Volume_Signal', 'SR_Signal', 'Trend_Signal', 'Momentum_Signal']
            
            # Remove NaN values and calculate composite
            valid_signals = signals[signal_columns].fillna(0)
            signals['Composite_Signal'] = valid_signals.mean(axis=1)
            
            # Generate buy/sell recommendations
            signals['Recommendation'] = self._generate_recommendations(signals['Composite_Signal'])
            
            # Calculate signal strength
            signals['Signal_Strength'] = abs(signals['Composite_Signal'])
            
            return signals
            
        except Exception as e:
            print(f"Signal generation error: {str(e)}")
            return None
    
    def _generate_ma_signals(self, data, indicators):
        """Generate moving average crossover signals"""
        try:
            if 'Moving_Averages' not in indicators:
                return pd.Series(0, index=data.index)
            
            ma_data = indicators['Moving_Averages']
            signals = pd.Series(0, index=data.index)
            
            # Short-term MA vs Long-term MA
            if 'MA_20' in ma_data.columns and 'MA_50' in ma_data.columns:
                ma_20 = ma_data['MA_20']
                ma_50 = ma_data['MA_50']
                
                # Golden cross (bullish) and death cross (bearish)
                golden_cross = (ma_20 > ma_50) & (ma_20.shift(1) <= ma_50.shift(1))
                death_cross = (ma_20 < ma_50) & (ma_20.shift(1) >= ma_50.shift(1))
                
                signals[golden_cross] = 1
                signals[death_cross] = -1
            
            return signals
            
        except Exception as e:
            print(f"MA signals error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _generate_rsi_signals(self, rsi):
        """Generate RSI-based signals"""
        try:
            if rsi is None:
                return pd.Series(0)
            
            signals = pd.Series(0, index=rsi.index)
            
            # Oversold/Overbought signals
            oversold = rsi < 30
            overbought = rsi > 70
            
            # RSI divergence (simplified)
            rsi_increasing = rsi > rsi.shift(1)
            rsi_decreasing = rsi < rsi.shift(1)
            
            # Generate signals
            signals[oversold & rsi_increasing] = 1  # Buy signal
            signals[overbought & rsi_decreasing] = -1  # Sell signal
            
            return signals
            
        except Exception as e:
            print(f"RSI signals error: {str(e)}")
            return pd.Series(0)
    
    def _generate_macd_signals(self, macd_data):
        """Generate MACD-based signals"""
        try:
            if macd_data is None or macd_data.empty:
                return pd.Series(0)
            
            signals = pd.Series(0, index=macd_data.index)
            
            # MACD crossover signals
            macd_line = macd_data['MACD']
            signal_line = macd_data['Signal']
            
            bullish_crossover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            bearish_crossover = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
            
            signals[bullish_crossover] = 1
            signals[bearish_crossover] = -1
            
            return signals
            
        except Exception as e:
            print(f"MACD signals error: {str(e)}")
            return pd.Series(0)
    
    def _generate_bollinger_signals(self, data, bb_data):
        """Generate Bollinger Bands signals"""
        try:
            if bb_data is None or bb_data.empty:
                return pd.Series(0, index=data.index)
            
            signals = pd.Series(0, index=data.index)
            
            price = data['Close']
            upper_band = bb_data['Upper']
            lower_band = bb_data['Lower']
            
            # Bollinger Band squeeze and expansion
            band_width = (upper_band - lower_band) / bb_data['Middle']
            narrow_bands = band_width < band_width.rolling(20).mean() * 0.8
            
            # Price touching bands
            touching_lower = price <= lower_band
            touching_upper = price >= upper_band
            
            # Generate signals
            signals[touching_lower & narrow_bands] = 1  # Buy at lower band during squeeze
            signals[touching_upper & narrow_bands] = -1  # Sell at upper band during squeeze
            
            return signals
            
        except Exception as e:
            print(f"Bollinger signals error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _generate_stochastic_signals(self, stoch_data):
        """Generate Stochastic oscillator signals"""
        try:
            if stoch_data is None or stoch_data.empty:
                return pd.Series(0)
            
            signals = pd.Series(0, index=stoch_data.index)
            
            k_percent = stoch_data['%K']
            d_percent = stoch_data['%D']
            
            # Stochastic crossover in oversold/overbought regions
            oversold_cross = (k_percent > d_percent) & (k_percent.shift(1) <= d_percent.shift(1)) & (k_percent < 20)
            overbought_cross = (k_percent < d_percent) & (k_percent.shift(1) >= d_percent.shift(1)) & (k_percent > 80)
            
            signals[oversold_cross] = 1
            signals[overbought_cross] = -1
            
            return signals
            
        except Exception as e:
            print(f"Stochastic signals error: {str(e)}")
            return pd.Series(0)
    
    def _generate_volume_signals(self, data):
        """Generate volume-based signals"""
        try:
            signals = pd.Series(0, index=data.index)
            
            volume = data['Volume']
            price = data['Close']
            
            # Volume moving average
            volume_ma = volume.rolling(20).mean()
            
            # Price and volume confirmation
            price_up = price > price.shift(1)
            price_down = price < price.shift(1)
            volume_up = volume > volume_ma * 1.5  # High volume
            
            # Volume confirmation signals
            signals[(price_up & volume_up)] = 0.5  # Bullish volume confirmation
            signals[(price_down & volume_up)] = -0.5  # Bearish volume confirmation
            
            return signals
            
        except Exception as e:
            print(f"Volume signals error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _generate_support_resistance_signals(self, data):
        """Generate support/resistance based signals"""
        try:
            signals = pd.Series(0, index=data.index)
            
            price = data['Close']
            high = data['High']
            low = data['Low']
            
            # Simple support/resistance using rolling min/max
            support = low.rolling(20).min()
            resistance = high.rolling(20).max()
            
            # Price breaking support/resistance
            breaking_resistance = (price > resistance.shift(1)) & (price.shift(1) <= resistance.shift(1))
            breaking_support = (price < support.shift(1)) & (price.shift(1) >= support.shift(1))
            
            signals[breaking_resistance] = 1  # Bullish breakout
            signals[breaking_support] = -1  # Bearish breakdown
            
            return signals
            
        except Exception as e:
            print(f"Support/Resistance signals error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _generate_trend_signals(self, data, indicators):
        """Generate trend-following signals"""
        try:
            signals = pd.Series(0, index=data.index)
            
            price = data['Close']
            
            # ADX for trend strength
            if 'ADX' in indicators:
                adx = indicators['ADX']
                strong_trend = adx > 25
                
                # Price above/below moving average in strong trend
                if 'Moving_Averages' in indicators and 'MA_50' in indicators['Moving_Averages'].columns:
                    ma_50 = indicators['Moving_Averages']['MA_50']
                    
                    uptrend = (price > ma_50) & strong_trend
                    downtrend = (price < ma_50) & strong_trend
                    
                    signals[uptrend] = 0.5
                    signals[downtrend] = -0.5
            
            return signals
            
        except Exception as e:
            print(f"Trend signals error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _generate_momentum_signals(self, indicators):
        """Generate momentum-based signals"""
        try:
            signals = pd.Series(0)
            
            # ROC (Rate of Change) signals
            if 'ROC' in indicators:
                roc = indicators['ROC']
                
                signals = pd.Series(0, index=roc.index)
                
                # Momentum acceleration/deceleration
                roc_increasing = roc > roc.shift(1)
                roc_decreasing = roc < roc.shift(1)
                positive_roc = roc > 0
                negative_roc = roc < 0
                
                signals[(positive_roc & roc_increasing)] = 0.3
                signals[(negative_roc & roc_decreasing)] = -0.3
            
            return signals
            
        except Exception as e:
            print(f"Momentum signals error: {str(e)}")
            return pd.Series(0)
    
    def _generate_recommendations(self, composite_signal):
        """Generate buy/sell/hold recommendations"""
        try:
            recommendations = pd.Series('HOLD', index=composite_signal.index)
            
            # Strong signals
            strong_buy = composite_signal > 0.6
            strong_sell = composite_signal < -0.6
            
            # Moderate signals
            buy = (composite_signal > 0.3) & (composite_signal <= 0.6)
            sell = (composite_signal < -0.3) & (composite_signal >= -0.6)
            
            recommendations[strong_buy] = 'STRONG BUY'
            recommendations[buy] = 'BUY'
            recommendations[sell] = 'SELL'
            recommendations[strong_sell] = 'STRONG SELL'
            
            return recommendations
            
        except Exception as e:
            print(f"Recommendations generation error: {str(e)}")
            return pd.Series('HOLD')
    
    def generate_alerts(self, data, signals, price_thresholds=None):
        """Generate trading alerts"""
        try:
            alerts = []
            current_price = data['Close'].iloc[-1]
            current_signal = signals['Composite_Signal'].iloc[-1]
            current_recommendation = signals['Recommendation'].iloc[-1]
            signal_strength = signals['Signal_Strength'].iloc[-1]
            
            # Signal-based alerts
            if current_recommendation in ['STRONG BUY', 'BUY']:
                alerts.append({
                    'type': 'BUY_SIGNAL',
                    'message': f"{current_recommendation} signal detected",
                    'strength': signal_strength,
                    'price': current_price,
                    'timestamp': data.index[-1]
                })
            elif current_recommendation in ['STRONG SELL', 'SELL']:
                alerts.append({
                    'type': 'SELL_SIGNAL',
                    'message': f"{current_recommendation} signal detected",
                    'strength': signal_strength,
                    'price': current_price,
                    'timestamp': data.index[-1]
                })
            
            # Price threshold alerts
            if price_thresholds:
                if 'upper_threshold' in price_thresholds and current_price >= price_thresholds['upper_threshold']:
                    alerts.append({
                        'type': 'PRICE_ALERT',
                        'message': f"Price reached upper threshold: ${price_thresholds['upper_threshold']:.2f}",
                        'price': current_price,
                        'timestamp': data.index[-1]
                    })
                
                if 'lower_threshold' in price_thresholds and current_price <= price_thresholds['lower_threshold']:
                    alerts.append({
                        'type': 'PRICE_ALERT',
                        'message': f"Price reached lower threshold: ${price_thresholds['lower_threshold']:.2f}",
                        'price': current_price,
                        'timestamp': data.index[-1]
                    })
            
            # Technical indicator alerts
            self._add_technical_alerts(data, alerts)
            
            return alerts
            
        except Exception as e:
            print(f"Alert generation error: {str(e)}")
            return []
    
    def _add_technical_alerts(self, data, alerts):
        """Add technical indicator based alerts"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # RSI alerts
            rsi = self.technical_analyzer.calculate_rsi(data['Close'])
            current_rsi = rsi.iloc[-1] if not rsi.empty else None
            
            if current_rsi:
                if current_rsi > 80:
                    alerts.append({
                        'type': 'RSI_ALERT',
                        'message': f"RSI extremely overbought: {current_rsi:.1f}",
                        'price': current_price,
                        'timestamp': data.index[-1]
                    })
                elif current_rsi < 20:
                    alerts.append({
                        'type': 'RSI_ALERT',
                        'message': f"RSI extremely oversold: {current_rsi:.1f}",
                        'price': current_price,
                        'timestamp': data.index[-1]
                    })
            
            # Volume spike alerts
            volume = data['Volume']
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            
            if current_volume > avg_volume * 3:
                alerts.append({
                    'type': 'VOLUME_ALERT',
                    'message': f"Unusual volume spike: {current_volume/avg_volume:.1f}x average",
                    'price': current_price,
                    'timestamp': data.index[-1]
                })
            
        except Exception as e:
            print(f"Technical alerts error: {str(e)}")
    
    def backtest_signals(self, data, signals, initial_capital=10000):
        """Backtest trading signals"""
        try:
            portfolio_value = initial_capital
            position = 0
            trades = []
            portfolio_history = []
            
            for i, (date, row) in enumerate(signals.iterrows()):
                if i == 0:
                    continue
                
                price = data.loc[date, 'Close']
                recommendation = row['Recommendation']
                
                # Execute trades based on recommendations
                if recommendation in ['BUY', 'STRONG BUY'] and position <= 0:
                    # Buy
                    shares_to_buy = portfolio_value / price
                    position += shares_to_buy
                    portfolio_value = 0
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'portfolio_value': position * price
                    })
                
                elif recommendation in ['SELL', 'STRONG SELL'] and position > 0:
                    # Sell
                    portfolio_value = position * price
                    position = 0
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': position,
                        'portfolio_value': portfolio_value
                    })
                
                # Calculate current portfolio value
                current_value = portfolio_value + (position * price)
                portfolio_history.append({
                    'date': date,
                    'portfolio_value': current_value,
                    'position': position,
                    'price': price
                })
            
            # Calculate final performance
            final_value = portfolio_history[-1]['portfolio_value'] if portfolio_history else initial_capital
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate buy and hold return for comparison
            initial_price = data['Close'].iloc[0]
            final_price = data['Close'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price
            
            return {
                'trades': trades,
                'portfolio_history': portfolio_history,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'final_value': final_value,
                'num_trades': len(trades)
            }
            
        except Exception as e:
            print(f"Backtesting error: {str(e)}")
            return None
    
    def calculate_risk_reward_ratio(self, signals, data, lookforward_days=30):
        """Calculate risk-reward ratio for signals"""
        try:
            ratios = []
            
            for i, (date, row) in enumerate(signals.iterrows()):
                if row['Recommendation'] in ['BUY', 'STRONG BUY']:
                    entry_price = data.loc[date, 'Close']
                    
                    # Look forward to calculate potential reward and risk
                    future_data = data.loc[date:].head(lookforward_days + 1)
                    
                    if len(future_data) > 1:
                        max_price = future_data['High'].max()
                        min_price = future_data['Low'].min()
                        
                        potential_reward = (max_price - entry_price) / entry_price
                        potential_risk = (entry_price - min_price) / entry_price
                        
                        if potential_risk > 0:
                            risk_reward_ratio = potential_reward / potential_risk
                            ratios.append({
                                'date': date,
                                'entry_price': entry_price,
                                'potential_reward': potential_reward,
                                'potential_risk': potential_risk,
                                'risk_reward_ratio': risk_reward_ratio
                            })
            
            return ratios
            
        except Exception as e:
            print(f"Risk-reward calculation error: {str(e)}")
            return []
