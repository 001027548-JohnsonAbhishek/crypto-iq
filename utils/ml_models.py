import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.predictions = {}
        
    def prepare_data(self, data, sequence_length=60):
        """Prepare data for ML models"""
        try:
            # Use closing prices
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
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model"""
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=True),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_lstm_model(self, data, sequence_length=60, epochs=50):
        """Train LSTM model"""
        try:
            X, y, scaled_data = self.prepare_data(data, sequence_length)
            if X is None:
                return None, None
            
            # Split data
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Build and train model
            model = self.build_lstm_model((X_train.shape[1], 1))
            history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, 
                              validation_data=(X_test, y_test), verbose=0)
            
            self.models['LSTM'] = model
            return model, history
            
        except Exception as e:
            print(f"LSTM training error: {str(e)}")
            return None, None
    
    def train_gru_model(self, data, sequence_length=60, epochs=50):
        """Train GRU model"""
        try:
            X, y, scaled_data = self.prepare_data(data, sequence_length)
            if X is None:
                return None, None
            
            # Split data
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Build and train model
            model = self.build_gru_model((X_train.shape[1], 1))
            history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, 
                              validation_data=(X_test, y_test), verbose=0)
            
            self.models['GRU'] = model
            return model, history
            
        except Exception as e:
            print(f"GRU training error: {str(e)}")
            return None, None
    
    def train_prophet_model(self, data):
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['Close'].values
            })
            
            # Create and fit model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            self.models['Prophet'] = model
            
            return model
            
        except Exception as e:
            print(f"Prophet training error: {str(e)}")
            return None
    
    def train_arima_model(self, data, order=(5,1,0)):
        """Train ARIMA model"""
        try:
            prices = data['Close'].values
            
            # Fit ARIMA model
            model = ARIMA(prices, order=order)
            fitted_model = model.fit()
            
            self.models['ARIMA'] = fitted_model
            return fitted_model
            
        except Exception as e:
            print(f"ARIMA training error: {str(e)}")
            return None
    
    def predict_lstm(self, data, days_ahead=30, sequence_length=60):
        """Make predictions using LSTM model"""
        try:
            if 'LSTM' not in self.models:
                return None
            
            model = self.models['LSTM']
            
            # Prepare last sequence
            last_sequence = data['Close'].values[-sequence_length:]
            last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
            
            predictions = []
            current_sequence = last_sequence_scaled.flatten()
            
            for _ in range(days_ahead):
                # Reshape for prediction
                X_pred = current_sequence[-sequence_length:].reshape(1, sequence_length, 1)
                
                # Make prediction
                pred_scaled = model.predict(X_pred, verbose=0)[0][0]
                predictions.append(pred_scaled)
                
                # Update sequence
                current_sequence = np.append(current_sequence, pred_scaled)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            return predictions
            
        except Exception as e:
            print(f"LSTM prediction error: {str(e)}")
            return None
    
    def predict_prophet(self, data, days_ahead=30):
        """Make predictions using Prophet model"""
        try:
            if 'Prophet' not in self.models:
                return None
            
            model = self.models['Prophet']
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Extract predictions for future dates
            predictions = forecast['yhat'].tail(days_ahead).values
            lower_bound = forecast['yhat_lower'].tail(days_ahead).values
            upper_bound = forecast['yhat_upper'].tail(days_ahead).values
            
            return predictions, lower_bound, upper_bound
            
        except Exception as e:
            print(f"Prophet prediction error: {str(e)}")
            return None
    
    def predict_arima(self, days_ahead=30):
        """Make predictions using ARIMA model"""
        try:
            if 'ARIMA' not in self.models:
                return None
            
            model = self.models['ARIMA']
            
            # Make forecast
            forecast = model.forecast(steps=days_ahead)
            conf_int = model.get_forecast(steps=days_ahead).conf_int()
            
            return forecast, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
            
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
            
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
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
            ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
            
            for model, prediction in predictions_dict.items():
                if model in weights:
                    ensemble_pred += weights[model] * prediction
            
            return ensemble_pred
            
        except Exception as e:
            print(f"Ensemble prediction error: {str(e)}")
            return None
    
    def detect_trend_changes(self, data, window=20):
        """Detect potential trend changes"""
        try:
            prices = data['Close']
            
            # Calculate moving averages
            short_ma = prices.rolling(window=window//2).mean()
            long_ma = prices.rolling(window=window).mean()
            
            # Detect crossovers
            bullish_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
            bearish_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
            
            return {
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'short_ma': short_ma,
                'long_ma': long_ma
            }
            
        except Exception as e:
            print(f"Trend detection error: {str(e)}")
            return None
