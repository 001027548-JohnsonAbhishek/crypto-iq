import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
from typing import Dict, List, Optional, Tuple
import time

class InstitutionalFlowTracker:
    """
    Track institutional investor flows in cryptocurrency markets
    Uses multiple data sources to identify large-scale trading patterns
    """
    
    def __init__(self):
        self.large_tx_threshold = 1000000  # $1M+ transactions considered institutional
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalyzer/1.0'
        })
    
    def get_exchange_flows(self, symbol: str, days: int = 30) -> Dict:
        """
        Analyze exchange inflows/outflows to identify institutional patterns
        """
        try:
            # Clean symbol and get historical data for volume analysis
            # Handle different symbol formats: BTC, $BTC, BTC-USD
            clean_symbol = symbol.replace('$', '').upper()
            if '-USD' not in clean_symbol:
                ticker_symbol = f"{clean_symbol}-USD"
            else:
                ticker_symbol = clean_symbol
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=f"{days}d", interval="1h")
            
            if data.empty:
                return {"error": "No data available"}
            
            # Calculate volume metrics
            avg_volume = data['Volume'].mean()
            volume_std = data['Volume'].std()
            
            # Identify institutional volume spikes (3+ standard deviations)
            institutional_threshold = avg_volume + (3 * volume_std)
            institutional_hours = data[data['Volume'] > institutional_threshold]
            
            # Calculate flow metrics
            total_volume = data['Volume'].sum()
            institutional_volume = institutional_hours['Volume'].sum()
            institutional_percentage = (institutional_volume / total_volume) * 100
            
            # Detect flow patterns
            recent_7d = data.tail(168)  # Last 7 days (hourly data)
            recent_institutional = recent_7d[recent_7d['Volume'] > institutional_threshold]
            
            return {
                "total_volume_24h": float(data.tail(24)['Volume'].sum()),
                "avg_volume": float(avg_volume),
                "institutional_threshold": float(institutional_threshold),
                "institutional_volume_percentage": float(institutional_percentage),
                "institutional_events_count": len(institutional_hours),
                "recent_institutional_activity": len(recent_institutional),
                "largest_volume_hour": {
                    "volume": float(data['Volume'].max()),
                    "timestamp": data['Volume'].idxmax().strftime("%Y-%m-%d %H:%M"),
                    "price_impact": float(data.loc[data['Volume'].idxmax(), 'Close'] - data.loc[data['Volume'].idxmax(), 'Open'])
                }
            }
            
        except Exception as e:
            return {"error": f"Exchange flow analysis failed: {str(e)}"}
    
    def get_whale_transactions(self, symbol: str) -> Dict:
        """
        Analyze large transactions that indicate institutional activity
        """
        try:
            # Clean symbol and simulate whale transaction detection based on volume patterns
            clean_symbol = symbol.replace('$', '').upper()
            if '-USD' not in clean_symbol:
                ticker_symbol = f"{clean_symbol}-USD"
            else:
                ticker_symbol = clean_symbol
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="7d", interval="15m")
            
            if data.empty:
                return {"error": "No whale data available"}
            
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate potential whale transactions based on volume spikes
            volume_mean = data['Volume'].mean()
            volume_std = data['Volume'].std()
            
            # Whale threshold: 5+ standard deviations above mean
            whale_threshold = volume_mean + (5 * volume_std)
            whale_transactions = data[data['Volume'] > whale_threshold]
            
            whale_analysis = []
            for idx, row in whale_transactions.iterrows():
                estimated_usd_value = row['Volume'] * row['Close']
                whale_analysis.append({
                    "timestamp": idx.strftime("%Y-%m-%d %H:%M"),
                    "volume": float(row['Volume']),
                    "price": float(row['Close']),
                    "estimated_usd_value": float(estimated_usd_value),
                    "price_change": float(row['Close'] - row['Open']),
                    "impact_score": float((row['Volume'] - volume_mean) / volume_std)
                })
            
            # Calculate whale flow metrics
            total_whale_volume = whale_transactions['Volume'].sum()
            avg_transaction_size = whale_transactions['Volume'].mean() if len(whale_transactions) > 0 else 0
            
            return {
                "whale_transactions_count": len(whale_transactions),
                "total_whale_volume": float(total_whale_volume),
                "avg_whale_transaction_size": float(avg_transaction_size),
                "whale_threshold_volume": float(whale_threshold),
                "largest_whale_transaction": float(whale_transactions['Volume'].max()) if len(whale_transactions) > 0 else 0,
                "recent_whale_activity": whale_analysis[-10:],  # Last 10 whale transactions
                "whale_flow_direction": self._calculate_flow_direction(whale_transactions)
            }
            
        except Exception as e:
            return {"error": f"Whale transaction analysis failed: {str(e)}"}
    
    def get_institutional_holdings(self, symbol: str) -> Dict:
        """
        Estimate institutional holdings and accumulation patterns
        """
        try:
            # Clean symbol and get longer term data for accumulation analysis
            clean_symbol = symbol.replace('$', '').upper()
            if '-USD' not in clean_symbol:
                ticker_symbol = f"{clean_symbol}-USD"
            else:
                ticker_symbol = clean_symbol
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                return {"error": "No holdings data available"}
            
            # Calculate accumulation metrics
            price_data = data['Close']
            volume_data = data['Volume']
            
            # Institutional accumulation indicators
            # 1. Volume-Weighted Average Price (VWAP) analysis
            vwap = (data['Close'] * data['Volume']).sum() / data['Volume'].sum()
            current_price = float(price_data.iloc[-1])
            
            # 2. Accumulation/Distribution Line
            money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
            money_flow_volume = money_flow_multiplier * data['Volume']
            accumulation_line = money_flow_volume.cumsum()
            
            # 3. On-Balance Volume (OBV) for institutional flow
            obv = []
            prev_obv = 0
            for i in range(len(data)):
                if i == 0:
                    obv.append(data['Volume'].iloc[i])
                    prev_obv = data['Volume'].iloc[i]
                else:
                    if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                        current_obv = prev_obv + data['Volume'].iloc[i]
                    elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                        current_obv = prev_obv - data['Volume'].iloc[i]
                    else:
                        current_obv = prev_obv
                    obv.append(current_obv)
                    prev_obv = current_obv
            
            # Recent trends (last 30 days)
            recent_data = data.tail(30)
            recent_obv = obv[-30:]
            
            obv_trend = "Accumulation" if recent_obv[-1] > recent_obv[0] else "Distribution"
            accumulation_trend = "Positive" if accumulation_line.iloc[-1] > accumulation_line.iloc[-30] else "Negative"
            
            return {
                "vwap": float(vwap),
                "current_price": current_price,
                "price_vs_vwap": float((current_price - vwap) / vwap * 100),
                "accumulation_distribution_trend": accumulation_trend,
                "obv_trend": obv_trend,
                "institutional_strength_score": self._calculate_institutional_strength(data, obv, accumulation_line),
                "estimated_institutional_percentage": self._estimate_institutional_percentage(data),
                "accumulation_periods": self._identify_accumulation_periods(data, obv),
                "distribution_periods": self._identify_distribution_periods(data, obv)
            }
            
        except Exception as e:
            return {"error": f"Institutional holdings analysis failed: {str(e)}"}
    
    def get_etf_flows(self, symbol: str) -> Dict:
        """
        Track ETF flows for institutional investment patterns
        """
        try:
            # Clean symbol and simulate ETF flow tracking based on volume and price patterns
            clean_symbol = symbol.replace('$', '').upper()
            if '-USD' not in clean_symbol:
                ticker_symbol = f"{clean_symbol}-USD"
            else:
                ticker_symbol = clean_symbol
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="30d", interval="1d")
            
            if data.empty:
                return {"error": "No ETF data available"}
            
            # Calculate potential ETF influence based on trading patterns
            # ETFs typically show consistent, large-volume trading
            daily_volumes = data['Volume']
            volume_consistency = 1 - (daily_volumes.std() / daily_volumes.mean())
            
            # Large consistent volumes suggest ETF activity
            avg_volume = daily_volumes.mean()
            high_volume_days = data[data['Volume'] > avg_volume * 1.5]
            
            # ETF flow estimation
            estimated_etf_days = len(high_volume_days)
            etf_flow_strength = volume_consistency * (estimated_etf_days / len(data))
            
            return {
                "estimated_etf_influence": float(etf_flow_strength),
                "volume_consistency_score": float(volume_consistency),
                "high_volume_trading_days": estimated_etf_days,
                "avg_daily_volume": float(avg_volume),
                "potential_etf_days": high_volume_days.index.strftime("%Y-%m-%d").tolist(),
                "etf_flow_trend": "Inflow" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "Outflow"
            }
            
        except Exception as e:
            return {"error": f"ETF flow analysis failed: {str(e)}"}
    
    def get_institutional_sentiment(self, symbol: str) -> Dict:
        """
        Analyze institutional sentiment indicators
        """
        try:
            clean_symbol = symbol.replace('$', '').upper()
            if '-USD' not in clean_symbol:
                ticker_symbol = f"{clean_symbol}-USD"
            else:
                ticker_symbol = clean_symbol
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="90d", interval="1d")
            
            if data.empty:
                return {"error": "No sentiment data available"}
            
            # Calculate sentiment indicators
            returns = data['Close'].pct_change().dropna()
            
            # Institutional sentiment metrics
            volatility = returns.std() * np.sqrt(365)
            sharpe_ratio = (returns.mean() * 365) / (returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
            
            # Price momentum (institutional follow-through)
            short_ma = data['Close'].rolling(window=10).mean()
            long_ma = data['Close'].rolling(window=50).mean()
            momentum_signal = "Bullish" if short_ma.iloc[-1] > long_ma.iloc[-1] else "Bearish"
            
            # Volume trend analysis
            volume_ma = data['Volume'].rolling(window=20).mean()
            recent_volume_trend = "Increasing" if data['Volume'].tail(5).mean() > volume_ma.iloc[-1] else "Decreasing"
            
            return {
                "institutional_sentiment": self._determine_sentiment_score(data),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "momentum_signal": momentum_signal,
                "volume_trend": recent_volume_trend,
                "price_stability_score": float(1 / (1 + volatility)),
                "institutional_confidence": self._calculate_confidence_score(data, returns)
            }
            
        except Exception as e:
            return {"error": f"Institutional sentiment analysis failed: {str(e)}"}
    
    def _calculate_flow_direction(self, transactions: pd.DataFrame) -> str:
        """Calculate the predominant flow direction from whale transactions"""
        if transactions.empty:
            return "Neutral"
        
        price_weighted_volume = (transactions['Close'] * transactions['Volume']).sum()
        total_volume = transactions['Volume'].sum()
        
        if price_weighted_volume > total_volume * transactions['Close'].mean():
            return "Bullish (Accumulation)"
        else:
            return "Bearish (Distribution)"
    
    def _calculate_institutional_strength(self, data: pd.DataFrame, obv: List, accumulation_line: pd.Series) -> float:
        """Calculate overall institutional strength score (0-100)"""
        try:
            # Volume consistency (institutions trade consistently)
            volume_consistency = 1 - (data['Volume'].std() / data['Volume'].mean())
            
            # Price stability during high volume (institutional absorption)
            high_vol_periods = data[data['Volume'] > data['Volume'].quantile(0.8)]
            price_stability = 1 - (high_vol_periods['Close'].std() / high_vol_periods['Close'].mean())
            
            # OBV trend strength
            obv_trend_strength = abs((obv[-1] - obv[0]) / obv[0]) if obv[0] != 0 else 0
            
            # Combine metrics
            strength_score = (volume_consistency * 0.4 + price_stability * 0.4 + min(obv_trend_strength, 1) * 0.2) * 100
            return float(min(max(strength_score, 0), 100))
            
        except:
            return 50.0  # Default neutral score
    
    def _estimate_institutional_percentage(self, data: pd.DataFrame) -> float:
        """Estimate what percentage of holdings are institutional"""
        try:
            # Based on volume patterns and price stability
            total_volume = data['Volume'].sum()
            high_volume_threshold = data['Volume'].quantile(0.9)
            institutional_volume = data[data['Volume'] > high_volume_threshold]['Volume'].sum()
            
            base_percentage = (institutional_volume / total_volume) * 100
            
            # Adjust based on market cap (larger market caps typically have higher institutional %)
            # This is a simplified estimation
            estimated_percentage = min(base_percentage * 2, 85)  # Cap at 85%
            return float(estimated_percentage)
            
        except:
            return 25.0  # Default estimate
    
    def _identify_accumulation_periods(self, data: pd.DataFrame, obv: List) -> List[Dict]:
        """Identify periods of institutional accumulation"""
        accumulation_periods = []
        
        try:
            # Look for periods where OBV increases while price is stable/declining
            for i in range(20, len(data) - 5):
                period_data = data.iloc[i-10:i+10]
                period_obv = obv[i-10:i+10]
                
                # Check if OBV is increasing while price is stable
                obv_trend = period_obv[-1] > period_obv[0]
                price_change = (period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) / period_data['Close'].iloc[0]
                
                if obv_trend and abs(price_change) < 0.05:  # Price relatively stable
                    accumulation_periods.append({
                        "start_date": period_data.index[0].strftime("%Y-%m-%d"),
                        "end_date": period_data.index[-1].strftime("%Y-%m-%d"),
                        "obv_change": float((period_obv[-1] - period_obv[0]) / period_obv[0] * 100),
                        "price_change": float(price_change * 100),
                        "strength": "High" if abs(price_change) < 0.02 else "Medium"
                    })
        except:
            pass
        
        return accumulation_periods[-5:]  # Return last 5 periods
    
    def _identify_distribution_periods(self, data: pd.DataFrame, obv: List) -> List[Dict]:
        """Identify periods of institutional distribution"""
        distribution_periods = []
        
        try:
            # Look for periods where OBV decreases while price is stable/increasing
            for i in range(20, len(data) - 5):
                period_data = data.iloc[i-10:i+10]
                period_obv = obv[i-10:i+10]
                
                # Check if OBV is decreasing
                obv_trend = period_obv[-1] < period_obv[0]
                price_change = (period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) / period_data['Close'].iloc[0]
                
                if obv_trend and price_change > -0.05:  # Price not falling too much
                    distribution_periods.append({
                        "start_date": period_data.index[0].strftime("%Y-%m-%d"),
                        "end_date": period_data.index[-1].strftime("%Y-%m-%d"),
                        "obv_change": float((period_obv[-1] - period_obv[0]) / period_obv[0] * 100),
                        "price_change": float(price_change * 100),
                        "strength": "High" if price_change > 0 else "Medium"
                    })
        except:
            pass
        
        return distribution_periods[-5:]  # Return last 5 periods
    
    def _determine_sentiment_score(self, data: pd.DataFrame) -> str:
        """Determine overall institutional sentiment"""
        try:
            # Price trend
            price_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]
            
            # Volume trend
            recent_volume = data['Volume'].tail(10).mean()
            avg_volume = data['Volume'].mean()
            volume_increase = recent_volume > avg_volume
            
            if price_trend > 0.1 and volume_increase:
                return "Very Bullish"
            elif price_trend > 0.05:
                return "Bullish"
            elif price_trend > -0.05:
                return "Neutral"
            elif price_trend > -0.1:
                return "Bearish"
            else:
                return "Very Bearish"
        except:
            return "Neutral"
    
    def _calculate_confidence_score(self, data: pd.DataFrame, returns: pd.Series) -> float:
        """Calculate institutional confidence score based on trading patterns"""
        try:
            # Consistent volume indicates confidence
            volume_cv = data['Volume'].std() / data['Volume'].mean()
            volume_score = max(0, 1 - volume_cv)
            
            # Lower volatility during high volume indicates institutional absorption
            high_volume_periods = data[data['Volume'] > data['Volume'].quantile(0.8)]
            if len(high_volume_periods) > 0:
                high_vol_volatility = high_volume_periods['Close'].pct_change().std()
                overall_volatility = returns.std()
                stability_score = max(0, 1 - (high_vol_volatility / overall_volatility))
            else:
                stability_score = 0.5
            
            confidence_score = (volume_score * 0.6 + stability_score * 0.4) * 100
            return float(min(max(confidence_score, 0), 100))
            
        except:
            return 50.0
    
    def get_comprehensive_institutional_analysis(self, symbol: str) -> Dict:
        """Get comprehensive institutional flow analysis"""
        try:
            exchange_flows = self.get_exchange_flows(symbol)
            whale_transactions = self.get_whale_transactions(symbol)
            institutional_holdings = self.get_institutional_holdings(symbol)
            etf_flows = self.get_etf_flows(symbol)
            institutional_sentiment = self.get_institutional_sentiment(symbol)
            
            # Calculate overall institutional activity score
            activity_score = self._calculate_overall_activity_score({
                'exchange_flows': exchange_flows,
                'whale_transactions': whale_transactions,
                'institutional_holdings': institutional_holdings,
                'etf_flows': etf_flows,
                'sentiment': institutional_sentiment
            })
            
            return {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_institutional_activity_score": activity_score,
                "exchange_flows": exchange_flows,
                "whale_transactions": whale_transactions,
                "institutional_holdings": institutional_holdings,
                "etf_flows": etf_flows,
                "institutional_sentiment": institutional_sentiment,
                "summary": self._generate_summary(activity_score, institutional_sentiment)
            }
            
        except Exception as e:
            return {"error": f"Comprehensive analysis failed: {str(e)}"}
    
    def _calculate_overall_activity_score(self, analyses: Dict) -> float:
        """Calculate overall institutional activity score (0-100)"""
        try:
            scores = []
            
            # Exchange flow score
            if 'institutional_volume_percentage' in analyses['exchange_flows']:
                scores.append(min(analyses['exchange_flows']['institutional_volume_percentage'], 100))
            
            # Whale activity score
            if 'whale_transactions_count' in analyses['whale_transactions']:
                whale_score = min(analyses['whale_transactions']['whale_transactions_count'] * 10, 100)
                scores.append(whale_score)
            
            # Holdings strength score
            if 'institutional_strength_score' in analyses['institutional_holdings']:
                scores.append(analyses['institutional_holdings']['institutional_strength_score'])
            
            # ETF influence score
            if 'estimated_etf_influence' in analyses['etf_flows']:
                scores.append(analyses['etf_flows']['estimated_etf_influence'] * 100)
            
            # Sentiment confidence score
            if 'institutional_confidence' in analyses['sentiment']:
                scores.append(analyses['sentiment']['institutional_confidence'])
            
            return float(sum(scores) / len(scores)) if scores else 50.0
            
        except:
            return 50.0
    
    def _generate_summary(self, activity_score: float, sentiment_data: Dict) -> str:
        """Generate human-readable summary of institutional activity"""
        try:
            if activity_score >= 80:
                activity_level = "Very High"
            elif activity_score >= 60:
                activity_level = "High"
            elif activity_score >= 40:
                activity_level = "Moderate"
            elif activity_score >= 20:
                activity_level = "Low"
            else:
                activity_level = "Very Low"
            
            sentiment = sentiment_data.get('institutional_sentiment', 'Neutral')
            
            return f"Institutional activity level: {activity_level} (Score: {activity_score:.1f}/100). Current sentiment: {sentiment}. This indicates {'strong' if activity_score >= 60 else 'moderate' if activity_score >= 40 else 'weak'} institutional presence in the market."
            
        except:
            return "Unable to generate summary due to insufficient data."