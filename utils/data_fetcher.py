import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import os

class CryptoDataFetcher:
    def __init__(self):
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.binance_api = "https://api.binance.com/api/v3"
        
    def get_crypto_data(self, symbol, days=365):
        """Fetch cryptocurrency data using multiple sources"""
        try:
            # Primary source: Yahoo Finance
            data = self._fetch_yfinance_data(symbol, days)
            if data is not None and not data.empty:
                return data
            
            # Fallback: Try alternative symbol format
            if '-USD' in symbol:
                alt_symbol = symbol.replace('-USD', 'USD')
                data = self._fetch_yfinance_data(alt_symbol, days)
                if data is not None and not data.empty:
                    return data
            
            # Additional fallback methods can be added here
            st.error(f"Could not fetch data for {symbol}")
            return None
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def _fetch_yfinance_data(self, symbol, days):
        """Fetch data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
                
            # Ensure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return None
            
            return data
            
        except Exception as e:
            print(f"YFinance error: {str(e)}")
            return None
    
    def get_crypto_info(self, symbol):
        """Get additional cryptocurrency information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except:
            return {}
    
    def get_market_data(self):
        """Get overall market data"""
        try:
            # Fetch major crypto market cap data
            url = f"{self.coingecko_api}/global"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_trending_cryptos(self):
        """Get trending cryptocurrencies"""
        try:
            url = f"{self.coingecko_api}/search/trending"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_fear_greed_index(self):
        """Get Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def validate_symbol(self, symbol):
        """Validate if a cryptocurrency symbol exists"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'regularMarketPrice' in info or 'previousClose' in info
        except:
            return False
    
    def get_real_time_price(self, symbol):
        """Get real-time price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except:
            return None
    
    def get_multiple_cryptos(self, symbols, days=365):
        """Fetch data for multiple cryptocurrencies"""
        results = {}
        for symbol in symbols:
            data = self.get_crypto_data(symbol, days)
            if data is not None:
                results[symbol] = data
        return results
    
    def calculate_correlation_matrix(self, symbols, days=365):
        """Calculate correlation matrix for multiple cryptocurrencies"""
        try:
            crypto_data = self.get_multiple_cryptos(symbols, days)
            
            if not crypto_data:
                return None
            
            # Extract closing prices
            price_data = pd.DataFrame()
            for symbol, data in crypto_data.items():
                price_data[symbol] = data['Close']
            
            # Calculate correlation matrix
            correlation_matrix = price_data.corr()
            return correlation_matrix
            
        except Exception as e:
            print(f"Correlation calculation error: {str(e)}")
            return None
