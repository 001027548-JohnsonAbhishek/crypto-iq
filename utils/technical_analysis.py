import pandas as pd
import numpy as np
import talib
from scipy import stats

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            rsi = talib.RSI(prices, timeperiod=period)
            return pd.Series(rsi, index=prices.index if hasattr(prices, 'index') else range(len(prices)))
        except:
            # Fallback calculation
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            macd, macdsignal, macdhist = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.DataFrame({
                'MACD': macd,
                'Signal': macdsignal,
                'Histogram': macdhist
            })
        except:
            # Fallback calculation
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return pd.DataFrame({
                'MACD': macd,
                'Signal': signal_line,
                'Histogram': histogram
            })
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return pd.DataFrame({
                'Upper': upper,
                'Middle': middle,
                'Lower': lower
            })
        except:
            # Fallback calculation
            prices_series = pd.Series(prices)
            sma = prices_series.rolling(period).mean()
            std = prices_series.rolling(period).std()
            
            return pd.DataFrame({
                'Upper': sma + (std * std_dev),
                'Middle': sma,
                'Lower': sma - (std * std_dev)
            })
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values, 
                                     fastk_period=k_period, slowk_period=3, slowd_period=d_period)
            return pd.DataFrame({
                '%K': slowk,
                '%D': slowd
            })
        except:
            # Fallback calculation
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(d_period).mean()
            
            return pd.DataFrame({
                '%K': k_percent,
                '%D': d_percent
            })
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(atr)
        except:
            # Fallback calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            return atr
    
    def calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        try:
            williams_r = talib.WILLR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(williams_r)
        except:
            # Fallback calculation
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r
    
    def calculate_cci(self, high, low, close, period=20):
        """Calculate Commodity Channel Index"""
        try:
            cci = talib.CCI(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(cci)
        except:
            # Fallback calculation
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(period).mean()
            mean_deviation = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci
    
    def calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index"""
        try:
            adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(adx)
        except:
            # Simplified fallback
            tr = self.calculate_atr(high, low, close, 1)
            plus_dm = (high.diff().where(high.diff() > low.diff().abs(), 0)).rolling(period).mean()
            minus_dm = (low.diff().abs().where(low.diff().abs() > high.diff(), 0)).rolling(period).mean()
            
            plus_di = 100 * (plus_dm / tr.rolling(period).mean())
            minus_di = 100 * (minus_dm / tr.rolling(period).mean())
            
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
            adx = dx.rolling(period).mean()
            return adx
    
    def calculate_momentum(self, prices, period=10):
        """Calculate Momentum"""
        try:
            momentum = talib.MOM(prices.values, timeperiod=period)
            return pd.Series(momentum)
        except:
            return prices.diff(period)
    
    def calculate_roc(self, prices, period=12):
        """Calculate Rate of Change"""
        try:
            roc = talib.ROC(prices.values, timeperiod=period)
            return pd.Series(roc)
        except:
            return ((prices / prices.shift(period)) - 1) * 100
    
    def calculate_moving_averages(self, prices, periods=[5, 10, 20, 50, 100, 200]):
        """Calculate multiple moving averages"""
        mas = {}
        for period in periods:
            try:
                mas[f'MA_{period}'] = prices.rolling(period).mean()
            except:
                pass
        return pd.DataFrame(mas)
    
    def calculate_ema(self, prices, periods=[12, 26, 50, 200]):
        """Calculate Exponential Moving Averages"""
        emas = {}
        for period in periods:
            try:
                emas[f'EMA_{period}'] = prices.ewm(span=period).mean()
            except:
                pass
        return pd.DataFrame(emas)
    
    def identify_support_resistance(self, prices, window=20, min_touches=2):
        """Identify support and resistance levels"""
        try:
            # Find local minima and maxima
            from scipy.signal import argrelextrema
            
            prices_array = prices.values
            
            # Find local minima (support)
            support_indices = argrelextrema(prices_array, np.less, order=window)[0]
            # Find local maxima (resistance)
            resistance_indices = argrelextrema(prices_array, np.greater, order=window)[0]
            
            support_levels = prices.iloc[support_indices]
            resistance_levels = prices.iloc[resistance_indices]
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'support_indices': support_indices,
                'resistance_indices': resistance_indices
            }
        except:
            return None
    
    def calculate_fibonacci_retracement(self, high_price, low_price):
        """Calculate Fibonacci retracement levels"""
        try:
            diff = high_price - low_price
            levels = {
                '0%': high_price,
                '23.6%': high_price - 0.236 * diff,
                '38.2%': high_price - 0.382 * diff,
                '50%': high_price - 0.5 * diff,
                '61.8%': high_price - 0.618 * diff,
                '78.6%': high_price - 0.786 * diff,
                '100%': low_price
            }
            return levels
        except:
            return None
    
    def detect_chart_patterns(self, prices, window=20):
        """Detect basic chart patterns"""
        try:
            patterns = {}
            
            # Double top/bottom detection
            high_points = prices.rolling(window).max()
            low_points = prices.rolling(window).min()
            
            # Head and shoulders pattern (simplified)
            patterns['head_and_shoulders'] = self._detect_head_and_shoulders(prices, window)
            
            # Triangle patterns
            patterns['triangles'] = self._detect_triangles(prices, window)
            
            return patterns
        except:
            return {}
    
    def _detect_head_and_shoulders(self, prices, window):
        """Simplified head and shoulders detection"""
        # This is a basic implementation - more sophisticated pattern recognition would be needed
        return []
    
    def _detect_triangles(self, prices, window):
        """Simplified triangle pattern detection"""
        # This is a basic implementation - more sophisticated pattern recognition would be needed
        return []
    
    def generate_signals(self, data):
        """Generate trading signals based on technical indicators"""
        signals = pd.DataFrame(index=data.index)
        
        # RSI signals
        rsi = self.calculate_rsi(data['Close'])
        signals['RSI_Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        
        # MACD signals
        macd_data = self.calculate_macd(data['Close'])
        signals['MACD_Signal'] = np.where(
            (macd_data['MACD'] > macd_data['Signal']) & 
            (macd_data['MACD'].shift(1) <= macd_data['Signal'].shift(1)), 1,
            np.where(
                (macd_data['MACD'] < macd_data['Signal']) & 
                (macd_data['MACD'].shift(1) >= macd_data['Signal'].shift(1)), -1, 0
            )
        )
        
        # Moving average signals
        ma_short = data['Close'].rolling(20).mean()
        ma_long = data['Close'].rolling(50).mean()
        signals['MA_Signal'] = np.where(ma_short > ma_long, 1, -1)
        
        # Bollinger Bands signals
        bb = self.calculate_bollinger_bands(data['Close'])
        signals['BB_Signal'] = np.where(
            data['Close'] < bb['Lower'], 1,
            np.where(data['Close'] > bb['Upper'], -1, 0)
        )
        
        # Composite signal
        signal_columns = ['RSI_Signal', 'MACD_Signal', 'MA_Signal', 'BB_Signal']
        signals['Composite_Signal'] = signals[signal_columns].mean(axis=1)
        
        return signals
    
    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            # Price-based indicators
            indicators['RSI'] = self.calculate_rsi(data['Close'])
            indicators['MACD'] = self.calculate_macd(data['Close'])
            indicators['Bollinger_Bands'] = self.calculate_bollinger_bands(data['Close'])
            indicators['Moving_Averages'] = self.calculate_moving_averages(data['Close'])
            indicators['EMA'] = self.calculate_ema(data['Close'])
            
            # Volume and volatility indicators
            indicators['ATR'] = self.calculate_atr(data['High'], data['Low'], data['Close'])
            
            # Oscillators
            indicators['Stochastic'] = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            indicators['Williams_R'] = self.calculate_williams_r(data['High'], data['Low'], data['Close'])
            indicators['CCI'] = self.calculate_cci(data['High'], data['Low'], data['Close'])
            
            # Trend indicators
            indicators['ADX'] = self.calculate_adx(data['High'], data['Low'], data['Close'])
            indicators['Momentum'] = self.calculate_momentum(data['Close'])
            indicators['ROC'] = self.calculate_roc(data['Close'])
            
            return indicators
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {}
