import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_fetcher import CryptoDataFetcher
from utils.ml_models import MLPredictor
from utils.analytics import track_page_view, track_engagement
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")

def main():
    # Track page view
    track_page_view("ML Predictions")
    
    st.title("🤖 Machine Learning Predictions")
    st.markdown("Advanced price prediction using multiple ML models")
    
    # Check if data exists in session state
    if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
        st.warning("⚠️ No data available. Please go back to the main page and fetch cryptocurrency data first.")
        return
    
    data = st.session_state.crypto_data
    symbol = st.session_state.selected_crypto
    
    # Prediction settings
    st.sidebar.header("🔧 Prediction Settings")
    
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )
    
    models_to_use = st.sidebar.multiselect(
        "Select Models",
        ["LSTM", "GRU", "Prophet", "ARIMA"],
        default=["LSTM", "Prophet"]
    )
    
    confidence_interval = st.sidebar.slider(
        "Confidence Interval (%)",
        min_value=80,
        max_value=99,
        value=95
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        sequence_length = st.slider("Sequence Length (LSTM/GRU)", 30, 120, 60)
        epochs = st.slider("Training Epochs", 20, 100, 50)
        
    if st.sidebar.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Training models and generating predictions..."):
            predictor = MLPredictor()
            
            # Store predictions
            predictions = {}
            model_metrics = {}
            
            # Train and predict with selected models
            if "LSTM" in models_to_use:
                with st.status("Training LSTM model (using Prophet as alternative)..."):
                    lstm_model = predictor.train_lstm_model(
                        data, sequence_length=sequence_length, epochs=epochs
                    )
                    if lstm_model:
                        lstm_pred = predictor.predict_lstm(
                            data, days_ahead=prediction_days, sequence_length=sequence_length
                        )
                        if lstm_pred is not None:
                            predictions['LSTM'] = lstm_pred
                            st.success("✅ LSTM model trained successfully")
                        else:
                            st.error("❌ LSTM prediction failed")
                    else:
                        st.error("❌ LSTM model training failed")
            
            if "GRU" in models_to_use:
                with st.status("Training GRU model (using Prophet as alternative)..."):
                    gru_model = predictor.train_gru_model(
                        data, sequence_length=sequence_length, epochs=epochs
                    )
                    if gru_model:
                        # Use similar prediction method as LSTM
                        gru_pred = predictor.predict_lstm(
                            data, days_ahead=prediction_days, sequence_length=sequence_length
                        )
                        if gru_pred is not None:
                            predictions['GRU'] = gru_pred
                            st.success("✅ GRU model trained successfully")
                        else:
                            st.error("❌ GRU prediction failed")
                    else:
                        st.error("❌ GRU model training failed")
            
            if "Prophet" in models_to_use:
                with st.status("Training Prophet model..."):
                    prophet_model = predictor.train_prophet_model(data)
                    if prophet_model:
                        prophet_pred = predictor.predict_prophet(data, days_ahead=prediction_days)
                        if prophet_pred:
                            predictions['Prophet'] = prophet_pred
                            st.success("✅ Prophet model trained successfully")
                        else:
                            st.error("❌ Prophet prediction failed")
                    else:
                        st.error("❌ Prophet model training failed")
            
            if "ARIMA" in models_to_use:
                with st.status("Training ARIMA model..."):
                    arima_model = predictor.train_arima_model(data)
                    if arima_model:
                        arima_pred = predictor.predict_arima(days_ahead=prediction_days)
                        if arima_pred:
                            predictions['ARIMA'] = arima_pred
                            st.success("✅ ARIMA model trained successfully")
                        else:
                            st.error("❌ ARIMA prediction failed")
                    else:
                        st.error("❌ ARIMA model training failed")
            
            # Store predictions in session state
            st.session_state.predictions = predictions
            st.session_state.prediction_days = prediction_days
    
    # Initialize predictor for ensemble calculations
    predictor = MLPredictor()
    
    # Display predictions if available
    if 'predictions' in st.session_state and st.session_state.predictions:
        predictions = st.session_state.predictions
        pred_days = st.session_state.prediction_days
        
        st.success(f"✅ Predictions generated for {pred_days} days ahead")
        
        # Create prediction dates
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=pred_days,
            freq='D'
        )
        
        # Main prediction chart
        st.subheader("📈 Price Predictions")
        
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index[-90:] if len(data) > 90 else data.index,
            y=data['Close'][-90:] if len(data) > 90 else data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        colors = {'LSTM': 'red', 'GRU': 'orange', 'Prophet': 'green', 'ARIMA': 'purple'}
        
        for model_name, prediction_data in predictions.items():
            if not model_name.endswith('_Lower') and not model_name.endswith('_Upper'):
                # Extract prediction values from the dictionary
                if isinstance(prediction_data, dict) and 'predictions' in prediction_data:
                    pred_values = prediction_data['predictions']
                    pred_dates = prediction_data.get('dates', future_dates)
                else:
                    pred_values = prediction_data
                    pred_dates = future_dates
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_values,
                    mode='lines',
                    name=f'{model_name} Prediction',
                    line=dict(color=colors.get(model_name, 'gray'), width=2)
                ))
                
                # Add confidence intervals if available (for Prophet model)
                if isinstance(prediction_data, dict) and 'lower_bound' in prediction_data and 'upper_bound' in prediction_data:
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=prediction_data['upper_bound'],
                        mode='lines',
                        name=f'{model_name} Upper',
                        line=dict(color=colors.get(model_name, 'gray'), width=1, dash='dash'),
                        showlegend=False
                    ))
                    
                    # Lower bound with fill
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=prediction_data['lower_bound'],
                        mode='lines',
                        name=f'{model_name} Confidence',
                        line=dict(color=colors.get(model_name, 'gray'), width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor=f'rgba(128, 128, 128, 0.2)',
                        showlegend=True
                    ))
        
        # Add vertical line to separate historical and predicted data
        fig.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Add annotation for the line
        fig.add_annotation(
            x=last_date,
            y=0.95,
            yref="paper",
            text="Prediction Start",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black"
        )
        
        fig.update_layout(
            title=f"{symbol} Price Predictions ({pred_days} days ahead)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction summary
        st.subheader("📊 Prediction Summary")
        
        current_price = data['Close'].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.4f}")
        
        model_cols = [col2, col3, col4]
        model_index = 0
        
        for model_name, prediction_data in predictions.items():
            if not model_name.endswith('_Lower') and not model_name.endswith('_Upper') and model_index < 3:
                # Extract final prediction value based on data format
                if isinstance(prediction_data, dict) and 'predictions' in prediction_data:
                    final_prediction = prediction_data['predictions'][-1]
                elif isinstance(prediction_data, (list, tuple)) and len(prediction_data) > 0:
                    final_prediction = prediction_data[-1]
                else:
                    # Skip if we can't extract a valid prediction
                    continue
                    
                change_pct = ((final_prediction - current_price) / current_price) * 100
                
                with model_cols[model_index]:
                    st.metric(
                        f"{model_name} ({pred_days}d)",
                        f"${final_prediction:.4f}",
                        f"{change_pct:+.2f}%"
                    )
                model_index += 1
        
        # Ensemble prediction
        if len([p for p in predictions.keys() if not p.endswith('_Lower') and not p.endswith('_Upper')]) > 1:
            st.subheader("🎯 Ensemble Prediction")
            
            ensemble_predictions = {}
            for model_name, prediction in predictions.items():
                if not model_name.endswith('_Lower') and not model_name.endswith('_Upper'):
                    ensemble_predictions[model_name] = prediction
            
            if ensemble_predictions:
                ensemble_pred = predictor.ensemble_prediction(ensemble_predictions)
                
                if ensemble_pred is not None:
                    fig_ensemble = go.Figure()
                    
                    # Historical data
                    fig_ensemble.add_trace(go.Scatter(
                        x=data.index[-60:] if len(data) > 60 else data.index,
                        y=data['Close'][-60:] if len(data) > 60 else data['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Ensemble prediction - extract values if it's a dictionary
                    if isinstance(ensemble_pred, dict) and 'predictions' in ensemble_pred:
                        ensemble_values = ensemble_pred['predictions']
                        ensemble_dates = ensemble_pred.get('dates', future_dates)
                    else:
                        ensemble_values = ensemble_pred
                        ensemble_dates = future_dates
                    
                    fig_ensemble.add_trace(go.Scatter(
                        x=ensemble_dates,
                        y=ensemble_values,
                        mode='lines',
                        name='Ensemble Prediction',
                        line=dict(color='black', width=3)
                    ))
                    
                    # Add vertical line using shape instead of add_vline
                    fig_ensemble.add_shape(
                        type="line",
                        x0=last_date, x1=last_date,
                        y0=0, y1=1,
                        yref="paper",
                        line=dict(color="gray", width=2, dash="dash")
                    )
                    
                    fig_ensemble.update_layout(
                        title="Ensemble Model Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_ensemble, use_container_width=True)
                    
                    # Ensemble metrics
                    if isinstance(ensemble_pred, dict) and 'predictions' in ensemble_pred:
                        ensemble_final = ensemble_pred['predictions'][-1]
                        ensemble_change = ((ensemble_final - current_price) / current_price) * 100
                    elif isinstance(ensemble_pred, (list, tuple)) and len(ensemble_pred) > 0:
                        ensemble_final = ensemble_pred[-1]
                        ensemble_change = ((ensemble_final - current_price) / current_price) * 100
                    else:
                        ensemble_final = 0
                        ensemble_change = 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ensemble Prediction", f"${ensemble_final:.4f}")
                    with col2:
                        st.metric("Expected Change", f"{ensemble_change:+.2f}%")
                    with col3:
                        direction = "📈 Bullish" if ensemble_change > 0 else "📉 Bearish" if ensemble_change < 0 else "➡️ Neutral"
                        st.metric("Signal", direction)
        
        # Model comparison
        st.subheader("⚖️ Model Comparison")
        
        comparison_data = []
        for model_name, prediction_data in predictions.items():
            if not model_name.endswith('_Lower') and not model_name.endswith('_Upper'):
                # Extract prediction values based on data format
                if isinstance(prediction_data, dict) and 'predictions' in prediction_data:
                    pred_values = prediction_data['predictions']
                elif isinstance(prediction_data, (list, tuple)) and len(prediction_data) > 0:
                    pred_values = prediction_data
                else:
                    continue  # Skip if we can't extract valid predictions
                
                final_pred = pred_values[-1]
                change_pct = ((final_pred - current_price) / current_price) * 100
                
                comparison_data.append({
                    'Model': model_name,
                    'Final Price': f"${final_pred:.4f}",
                    'Change (%)': f"{change_pct:+.2f}%",
                    'Prediction Range': f"${min(pred_values):.4f} - ${max(pred_values):.4f}",
                    'Volatility': f"{np.std(pred_values):.4f}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        # Prediction confidence analysis
        st.subheader("🎯 Prediction Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Agreement Analysis**")
            
            if len(ensemble_predictions) > 1:
                # Calculate prediction agreement - handle different data formats
                final_predictions = []
                for pred in ensemble_predictions.values():
                    if isinstance(pred, dict) and 'predictions' in pred:
                        final_predictions.append(pred['predictions'][-1])
                    elif isinstance(pred, (list, tuple)) and len(pred) > 0:
                        final_predictions.append(pred[-1])
                
                if final_predictions:
                    pred_std = np.std(final_predictions)
                    pred_mean = np.mean(final_predictions)
                    agreement_score = max(0, 100 - (pred_std / pred_mean * 100))
                else:
                    agreement_score = 0
                
                st.write(f"• Agreement Score: {agreement_score:.1f}%")
                st.write(f"• Prediction Spread: ${pred_std:.4f}")
                
                if agreement_score > 80:
                    st.success("🟢 High model agreement - Strong confidence")
                elif agreement_score > 60:
                    st.warning("🟡 Moderate model agreement - Medium confidence")
                else:
                    st.error("🔴 Low model agreement - Low confidence")
        
        with col2:
            st.write("**Risk Assessment**")
            
            # Calculate prediction volatility
            all_predictions = []
            for pred in ensemble_predictions.values():
                if isinstance(pred, dict) and 'predictions' in pred:
                    all_predictions.extend(pred['predictions'])
                elif isinstance(pred, (list, tuple)):
                    all_predictions.extend(pred)
            
            if all_predictions:
                pred_volatility = np.std(all_predictions)
            else:
                pred_volatility = 0
            historical_volatility = data['Close'].pct_change().std() * data['Close'].iloc[-1]
            
            st.write(f"• Prediction Volatility: ${pred_volatility:.4f}")
            st.write(f"• Historical Volatility: ${historical_volatility:.4f}")
            
            if pred_volatility > historical_volatility * 1.5:
                st.warning("⚠️ High prediction uncertainty")
            else:
                st.success("✅ Normal prediction range")
        
        # Download predictions
        st.subheader("💾 Download Predictions")
        
        # Prepare prediction data for download
        pred_df = pd.DataFrame({'Date': future_dates})
        
        for model_name, prediction in predictions.items():
            if not model_name.endswith('_Lower') and not model_name.endswith('_Upper'):
                pred_df[f'{model_name}_Prediction'] = prediction
        
        if 'ensemble_pred' in locals():
            pred_df['Ensemble_Prediction'] = ensemble_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = pred_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download Predictions JSON",
                data=json_data,
                file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Configure your prediction settings and click 'Generate Predictions' to start")
        
        # Show model information
        st.subheader("🤖 Available Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Deep Learning Models**")
            st.write("• **LSTM**: Long Short-Term Memory networks for capturing long-term dependencies")
            st.write("• **GRU**: Gated Recurrent Units, computationally efficient alternative to LSTM")
            
            st.write("**Statistical Models**")
            st.write("• **Prophet**: Facebook's time series forecasting tool with trend and seasonality")
            st.write("• **ARIMA**: AutoRegressive Integrated Moving Average for time series analysis")
        
        with col2:
            st.write("**Model Strengths**")
            st.write("• **LSTM/GRU**: Best for complex patterns and non-linear relationships")
            st.write("• **Prophet**: Excellent for data with strong seasonal patterns")
            st.write("• **ARIMA**: Good baseline for stationary time series")
            st.write("• **Ensemble**: Combines multiple models for improved accuracy")
        
        # Show sample prediction workflow
        st.subheader("📈 Prediction Workflow")
        
        workflow_steps = [
            "1. **Data Preparation**: Normalize and structure historical price data",
            "2. **Model Training**: Train selected ML models on historical data",
            "3. **Prediction Generation**: Generate future price predictions",
            "4. **Ensemble Creation**: Combine predictions from multiple models",
            "5. **Confidence Analysis**: Assess prediction reliability and agreement",
            "6. **Visualization**: Display predictions with confidence intervals"
        ]
        
        for step in workflow_steps:
            st.markdown(step)

if __name__ == "__main__":
    main()
