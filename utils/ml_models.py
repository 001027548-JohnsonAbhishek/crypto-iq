import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# TensorFlow disabled due to compatibility issues - using Prophet and ARIMA instead
TENSORFLOW_AVAILABLE = False

class MLPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.predictions = {}
        
    def prepare_data(self, data, sequence_length=60):
        """Prepare data for ML models"""
        try:
            if len(data) < sequence_length:
                return None, None, None
            
            prices = data['Close'].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(prices)
            
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y, scaled_data
        except Exception as e:
            print(f"Data preparation error: {str(e)}")
            return None, None, None
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model - TensorFlow not available"""
        return None
    
    def build_gru_model(self, input_shape):
        """Build GRU model - TensorFlow not available"""
        return None
    
    def train_lstm_model(self, data, sequence_length=60, epochs=50):
        """Train LSTM model - using Prophet as alternative"""
        try:
            return self.train_prophet_model(data)
        except Exception as e:
            print(f"LSTM training error: {str(e)}")
            return None
    
    def train_gru_model(self, data, sequence_length=60, epochs=50):
        """Train GRU model - using Prophet as alternative"""
        try:
            return self.train_prophet_model(data)
        except Exception as e:
            print(f"GRU training error: {str(e)}")
            return None
    
    def train_prophet_model(self, data):
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['Close'].values
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            self.models['prophet'] = model
            
            return model
            
        except Exception as e:
            print(f"Prophet training error: {str(e)}")
            return None
    
    def train_arima_model(self, data, order=(5,1,0)):
        """Train ARIMA model"""
        try:
            prices = data['Close']
            
            # Fit ARIMA model
            model = ARIMA(prices, order=order)
            fitted_model = model.fit()
            
            self.models['arima'] = fitted_model
            return fitted_model
            
        except Exception as e:
            print(f"ARIMA training error: {str(e)}")
            return None
    
    def predict_lstm(self, data, days_ahead=30, sequence_length=60):
        """Make predictions using LSTM model - using Prophet as alternative"""
        return self.predict_prophet(data, days_ahead)
    
    def predict_prophet(self, data, days_ahead=30):
        """Make predictions using Prophet model"""
        try:
            if 'prophet' not in self.models:
                model = self.train_prophet_model(data)
                if model is None:
                    return None
            else:
                model = self.models['prophet']
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Extract predictions
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead)
            
            return {
                'dates': predictions['ds'].tolist(),
                'predictions': predictions['yhat'].tolist(),
                'lower_bound': predictions['yhat_lower'].tolist(),
                'upper_bound': predictions['yhat_upper'].tolist(),
                'model_type': 'Prophet'
            }
            
        except Exception as e:
            print(f"Prophet prediction error: {str(e)}")
            return None
    
    def predict_arima(self, days_ahead=30):
        """Make predictions using ARIMA model"""
        try:
            if 'arima' not in self.models:
                return None
            
            model = self.models['arima']
            forecast = model.forecast(steps=days_ahead)
            
            # Create future dates
            last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else pd.Timestamp.now()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=days_ahead, freq='D')
            
            return {
                'dates': future_dates.tolist(),
                'predictions': forecast.tolist(),
                'model_type': 'ARIMA'
            }
            
        except Exception as e:
            print(f"ARIMA prediction error: {str(e)}")
            return None
    
    def calculate_accuracy_metrics(self, actual, predicted):
        """Calculate model accuracy metrics"""
        try:
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, predicted)
            
            # Calculate percentage accuracy
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            print(f"Metrics calculation error: {str(e)}")
            return None
    
    def ensemble_prediction(self, predictions_dict, weights=None):
        """Create ensemble prediction from multiple models"""
        try:
            if not predictions_dict:
                return None
            
            # Default equal weights
            if weights is None:
                weights = {model: 1/len(predictions_dict) for model in predictions_dict}
            
            # Calculate weighted average
            ensemble_pred = np.zeros(len(list(predictions_dict.values())[0]['predictions']))
            
            for model_name, pred in predictions_dict.items():
                weight = weights.get(model_name, 0)
                ensemble_pred += np.array(pred['predictions']) * weight
            
            # Use dates from first model
            first_model = list(predictions_dict.values())[0]
            
            return {
                'dates': first_model['dates'],
                'predictions': ensemble_pred.tolist(),
                'model_type': 'Ensemble'
            }
            
        except Exception as e:
            print(f"Ensemble prediction error: {str(e)}")
            return None
    
    def detect_trend_changes(self, data, window=20):
        """Detect potential trend changes"""
        try:
            prices = data['Close']
            
            # Calculate moving averages
            ma_short = prices.rolling(window//2).mean()
            ma_long = prices.rolling(window).mean()
            
            # Detect crossovers
            crossovers = []
            for i in range(1, len(ma_short)):
                if ma_short.iloc[i-1] <= ma_long.iloc[i-1] and ma_short.iloc[i] > ma_long.iloc[i]:
                    crossovers.append({
                        'date': ma_short.index[i],
                        'type': 'bullish_crossover',
                        'price': prices.iloc[i]
                    })
                elif ma_short.iloc[i-1] >= ma_long.iloc[i-1] and ma_short.iloc[i] < ma_long.iloc[i]:
                    crossovers.append({
                        'date': ma_short.index[i],
                        'type': 'bearish_crossover',
                        'price': prices.iloc[i]
                    })
            
            return crossovers
            
        except Exception as e:
            print(f"Trend change detection error: {str(e)}")
            return []