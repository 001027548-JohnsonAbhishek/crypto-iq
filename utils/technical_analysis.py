import pandas as pd
import numpy as np
from scipy import stats

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
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
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        sma = prices_series.rolling(period).mean()
        std = prices_series.rolling(period).std()
        
        return pd.DataFrame({
            'Upper': sma + (std * std_dev),
            'Middle': sma,
            'Lower': sma - (std * std_dev)
        })
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
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
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    def calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_cci(self, high, low, close, period=20):
        """Calculate Commodity Channel Index"""
        tp = (high + low + close) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (tp - ma) / (0.015 * md)
        return cci
    
    def calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index"""
        # Calculate directional movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Only keep positive movements
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, 1)
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus.rolling(period).sum() / tr.rolling(period).sum())
        di_minus = 100 * (dm_minus.rolling(period).sum() / tr.rolling(period).sum())
        
        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.rolling(period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'DI+': di_plus,
            'DI-': di_minus
        })
    
    def calculate_momentum(self, prices, period=10):
        """Calculate Momentum"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        return prices_series.diff(period)
    
    def calculate_roc(self, prices, period=12):
        """Calculate Rate of Change"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        return ((prices_series / prices_series.shift(period)) - 1) * 100
    
    def calculate_moving_averages(self, prices, periods=[5, 10, 20, 50, 100, 200]):
        """Calculate multiple moving averages"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        mas = {}
        
        for period in periods:
            mas[f'MA_{period}'] = prices_series.rolling(period).mean()
        
        return pd.DataFrame(mas)
    
    def calculate_ema(self, prices, periods=[12, 26, 50, 200]):
        """Calculate Exponential Moving Averages"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        emas = {}
        
        for period in periods:
            emas[f'EMA_{period}'] = prices_series.ewm(span=period).mean()
        
        return pd.DataFrame(emas)
    
    def identify_support_resistance(self, prices, window=20, min_touches=2):
        """Identify support and resistance levels"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        
        # Find local maxima and minima
        highs = prices_series.rolling(window=window, center=True).max()
        lows = prices_series.rolling(window=window, center=True).min()
        
        resistance_levels = prices_series[prices_series == highs].dropna()
        support_levels = prices_series[prices_series == lows].dropna()
        
        return {
            'support_levels': support_levels.tolist(),
            'resistance_levels': resistance_levels.tolist()
        }
    
    def calculate_fibonacci_retracement(self, high_price, low_price):
        """Calculate Fibonacci retracement levels"""
        diff = high_price - low_price
        
        levels = {
            '0%': high_price,
            '23.6%': high_price - (0.236 * diff),
            '38.2%': high_price - (0.382 * diff),
            '50%': high_price - (0.5 * diff),
            '61.8%': high_price - (0.618 * diff),
            '100%': low_price
        }
        
        return levels
    
    def detect_chart_patterns(self, prices, window=20):
        """Detect basic chart patterns"""
        prices_series = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        
        patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders(prices_series, window),
            'triangle': self._detect_triangles(prices_series, window)
        }
        
        return patterns
    
    def _detect_head_and_shoulders(self, prices, window):
        """Simplified head and shoulders detection"""
        # This is a basic implementation
        # In practice, this would require more sophisticated pattern recognition
        rolling_max = prices.rolling(window).max()
        rolling_min = prices.rolling(window).min()
        
        # Look for three peaks pattern
        peaks = prices[prices == rolling_max].dropna()
        
        if len(peaks) >= 3:
            return "Potential head and shoulders pattern detected"
        return "No clear pattern"
    
    def _detect_triangles(self, prices, window):
        """Simplified triangle pattern detection"""
        # Basic triangle detection based on converging trend lines
        recent_prices = prices.tail(window)
        
        highs = recent_prices.rolling(5).max()
        lows = recent_prices.rolling(5).min()
        
        # Check if range is narrowing (triangle pattern)
        range_start = highs.iloc[0] - lows.iloc[0]
        range_end = highs.iloc[-1] - lows.iloc[-1]
        
        if range_end < range_start * 0.8:
            return "Potential triangle pattern detected"
        return "No clear pattern"
    
    def generate_signals(self, data):
        """Generate trading signals based on technical indicators"""
        signals = pd.DataFrame(index=data.index)
        
        # RSI signals
        rsi = self.calculate_rsi(data['Close'])
        signals['RSI_Signal'] = 0
        signals.loc[rsi < 30, 'RSI_Signal'] = 1  # Buy signal
        signals.loc[rsi > 70, 'RSI_Signal'] = -1  # Sell signal
        
        # MACD signals
        macd_data = self.calculate_macd(data['Close'])
        signals['MACD_Signal'] = 0
        signals.loc[macd_data['MACD'] > macd_data['Signal'], 'MACD_Signal'] = 1
        signals.loc[macd_data['MACD'] < macd_data['Signal'], 'MACD_Signal'] = -1
        
        # Moving average signals
        ma_short = data['Close'].rolling(20).mean()
        ma_long = data['Close'].rolling(50).mean()
        signals['MA_Signal'] = 0
        signals.loc[ma_short > ma_long, 'MA_Signal'] = 1
        signals.loc[ma_short < ma_long, 'MA_Signal'] = -1
        
        return signals
    
    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        indicators = {}
        
        try:
            indicators['RSI'] = self.calculate_rsi(data['Close'])
            indicators['MACD'] = self.calculate_macd(data['Close'])
            indicators['BB'] = self.calculate_bollinger_bands(data['Close'])
            indicators['Stoch'] = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            indicators['ATR'] = self.calculate_atr(data['High'], data['Low'], data['Close'])
            indicators['Williams_R'] = self.calculate_williams_r(data['High'], data['Low'], data['Close'])
            indicators['CCI'] = self.calculate_cci(data['High'], data['Low'], data['Close'])
            indicators['ADX'] = self.calculate_adx(data['High'], data['Low'], data['Close'])
            indicators['Momentum'] = self.calculate_momentum(data['Close'])
            indicators['ROC'] = self.calculate_roc(data['Close'])
            indicators['MA'] = self.calculate_moving_averages(data['Close'])
            indicators['EMA'] = self.calculate_ema(data['Close'])
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return indicators