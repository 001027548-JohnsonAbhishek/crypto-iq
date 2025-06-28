# Crypto Analysis Platform

## Overview

This is a comprehensive cryptocurrency analysis platform built with Streamlit that provides advanced technical analysis, machine learning predictions, sentiment analysis, and portfolio optimization capabilities. The application offers a multi-page interface for analyzing cryptocurrency data with various analytical tools and visualization capabilities.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application
- **Layout**: Multi-page application with sidebar navigation
- **Visualization**: Plotly for interactive charts and graphs
- **UI Components**: Streamlit widgets for user interaction and configuration

### Backend Architecture
- **Data Processing**: Pandas for data manipulation and analysis
- **Machine Learning**: TensorFlow/Keras for LSTM/GRU models, Prophet for time series forecasting
- **Technical Analysis**: TA-Lib library with custom fallback implementations
- **Portfolio Optimization**: SciPy optimization algorithms implementing Modern Portfolio Theory

### Data Sources
- **Primary**: Yahoo Finance API (yfinance) for historical price data
- **Social Media**: Twitter API v2 and Reddit API for sentiment analysis
- **News**: Configurable news API integration for sentiment analysis
- **Fallback**: Multiple data source redundancy for reliability

## Key Components

### Core Application (`app.py`)
- Main application entry point with cryptocurrency selection
- Session state management for data persistence
- Popular cryptocurrency symbols mapping
- Configuration sidebar for user preferences

### Analysis Pages
1. **Price Analysis** (`pages/1_Price_Analysis.py`)
   - Multiple chart types (Candlestick, Line, Area, OHLC)
   - Timeframe analysis options
   - Volume analysis integration
   - Technical indicator overlays

2. **ML Predictions** (`pages/2_ML_Predictions.py`)
   - LSTM, GRU, Prophet, and ARIMA model implementations
   - Configurable prediction horizons (7-90 days)
   - Confidence interval calculations
   - Model performance metrics

3. **Technical Analysis** (`pages/3_Technical_Analysis.py`)
   - Comprehensive technical indicator suite
   - Trend, momentum, and volatility indicators
   - Custom indicator parameter configuration
   - Pattern recognition capabilities

4. **Sentiment Analysis** (`pages/4_Sentiment_Analysis.py`)
   - Multi-source sentiment aggregation
   - Twitter, Reddit, and news sentiment analysis
   - TextBlob and VADER sentiment engines
   - Social media data filtering and processing

5. **Portfolio Optimization** (`pages/5_Portfolio_Optimization.py`)
   - Modern Portfolio Theory implementation
   - Efficient frontier calculation
   - Risk-return optimization
   - Multi-asset portfolio allocation

6. **Trading Signals** (`pages/6_Trading_Signals.py`)
   - Multi-indicator signal generation
   - Signal strength and confidence scoring
   - Backtesting capabilities
   - Automated alert system foundation

### Utility Modules

#### Data Fetcher (`utils/data_fetcher.py`)
- Multi-source data acquisition with fallback mechanisms
- Yahoo Finance primary integration
- Data validation and cleaning
- Error handling and retry logic

#### ML Models (`utils/ml_models.py`)
- Deep learning model implementations (LSTM, GRU)
- Time series forecasting (Prophet, ARIMA)
- Feature engineering and data preprocessing
- Model evaluation metrics

#### Technical Analysis (`utils/technical_analysis.py`)
- TA-Lib integration with custom fallbacks
- Comprehensive indicator calculations
- RSI, MACD, Bollinger Bands, and other popular indicators
- Statistical analysis functions

#### Portfolio Optimizer (`utils/portfolio_optimizer.py`)
- Modern Portfolio Theory calculations
- Risk metrics (VaR, Sharpe ratio, volatility)
- Efficient frontier optimization
- Monte Carlo simulation capabilities

#### Sentiment Analysis (`utils/sentiment_analysis.py`)
- Multi-platform social media integration
- TextBlob and VADER sentiment analysis
- API credential management
- Text preprocessing and cleaning

#### Trading Signals (`utils/trading_signals.py`)
- Multi-indicator signal aggregation
- Signal strength calculation
- Historical signal tracking
- Pattern recognition algorithms

## Data Flow

1. **Data Acquisition**: User selects cryptocurrency → Data fetcher retrieves historical data from Yahoo Finance
2. **Data Storage**: Raw data stored in Streamlit session state for cross-page access
3. **Analysis Processing**: Each page processes stored data through respective utility modules
4. **Visualization**: Processed results displayed through Plotly interactive charts
5. **User Interaction**: Real-time parameter updates trigger re-analysis and visualization updates

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization

### Data Sources
- **yfinance**: Yahoo Finance API client
- **requests**: HTTP requests for API calls

### Machine Learning
- **tensorflow**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **prophet**: Time series forecasting
- **statsmodels**: Statistical analysis

### Technical Analysis
- **talib**: Technical analysis library

### Sentiment Analysis
- **tweepy**: Twitter API client
- **praw**: Reddit API client
- **textblob**: Natural language processing
- **vaderSentiment**: Sentiment analysis

### Portfolio Optimization
- **scipy**: Scientific computing and optimization

## Deployment Strategy

### Environment Setup
- Python 3.8+ required for all dependencies
- Virtual environment recommended for isolation
- Environment variables for API credentials

### Configuration Requirements
- Twitter Bearer Token (optional, for sentiment analysis)
- Reddit API credentials (optional, for sentiment analysis)
- News API key (optional, for sentiment analysis)

### Scalability Considerations
- Session state management for multi-user environments
- API rate limiting and caching strategies
- Data persistence options for production deployment

### Performance Optimization
- Lazy loading of ML models
- Caching of technical indicators
- Asynchronous data fetching capabilities

## Changelog

- June 28, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.