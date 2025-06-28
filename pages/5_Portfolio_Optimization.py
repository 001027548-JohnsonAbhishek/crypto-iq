import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.data_fetcher import CryptoDataFetcher
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio Optimization", page_icon="💼", layout="wide")

def main():
    st.title("💼 Portfolio Optimization & Risk Analysis")
    st.markdown("Modern Portfolio Theory implementation for cryptocurrency portfolios")
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer()
    fetcher = CryptoDataFetcher()
    
    # Sidebar for portfolio configuration
    st.sidebar.header("⚙️ Portfolio Configuration")
    
    # Portfolio assets selection
    st.sidebar.subheader("Select Assets")
    
    # Popular crypto assets
    popular_cryptos = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Binance Coin': 'BNB-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD',
        'XRP': 'XRP-USD',
        'Polkadot': 'DOT-USD',
        'Chainlink': 'LINK-USD',
        'Litecoin': 'LTC-USD',
        'Polygon': 'MATIC-USD'
    }
    
    selected_assets = st.sidebar.multiselect(
        "Choose cryptocurrencies for your portfolio",
        options=list(popular_cryptos.keys()),
        default=['Bitcoin', 'Ethereum', 'Cardano', 'Solana'],
        help="Select 3-10 assets for optimal diversification"
    )
    
    # Convert to symbols
    selected_symbols = [popular_cryptos[asset] for asset in selected_assets]
    
    # Custom asset input
    custom_symbols = st.sidebar.text_input(
        "Add custom symbols (comma-separated)",
        placeholder="LINK-USD, DOT-USD",
        help="Enter additional crypto symbols"
    )
    
    if custom_symbols:
        custom_list = [s.strip().upper() for s in custom_symbols.split(',')]
        selected_symbols.extend(custom_list)
        selected_assets.extend([s.split('-')[0] for s in custom_list])
    
    # Time period for analysis
    time_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1y", "2y", "3y", "5y"],
        index=1,
        help="Longer periods provide more stable optimization results"
    )
    
    # Optimization method
    optimization_method = st.sidebar.selectbox(
        "Optimization Method",
        ["sharpe", "min_variance"],
        format_func=lambda x: "Maximize Sharpe Ratio" if x == "sharpe" else "Minimize Variance"
    )
    
    # Risk parameters
    st.sidebar.subheader("Risk Parameters")
    
    risk_free_rate = st.sidebar.slider(
        "Risk-free Rate (%)",
        0.0, 5.0, 2.0, 0.1,
        help="Government bond yield or safe asset return"
    ) / 100
    
    confidence_level = st.sidebar.slider(
        "VaR Confidence Level (%)",
        90, 99, 95, 1
    ) / 100
    
    # Portfolio analysis
    if len(selected_symbols) >= 2:
        if st.sidebar.button("🔍 Optimize Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio..."):
                # Update risk-free rate
                optimizer.risk_free_rate = risk_free_rate
                
                # Optimize portfolio
                optimization_result = optimizer.optimize_portfolio(
                    selected_symbols, 
                    optimization_method=optimization_method
                )
                
                if optimization_result and optimization_result['optimization_success']:
                    st.session_state.optimization_result = optimization_result
                    st.session_state.portfolio_symbols = selected_symbols
                    st.session_state.portfolio_assets = selected_assets
                    st.success("✅ Portfolio optimization completed successfully!")
                else:
                    st.error("❌ Portfolio optimization failed. Please check your asset selection.")
    else:
        st.sidebar.warning("⚠️ Select at least 2 assets for portfolio optimization")
    
    # Display optimization results
    if 'optimization_result' in st.session_state:
        result = st.session_state.optimization_result
        symbols = st.session_state.portfolio_symbols
        assets = st.session_state.portfolio_assets
        
        # Portfolio Summary
        st.subheader("📊 Optimized Portfolio Summary")
        
        portfolio_metrics = result['portfolio_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Annual Return",
                f"{portfolio_metrics['return']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Annual Volatility",
                f"{portfolio_metrics['volatility']*100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{portfolio_metrics['sharpe_ratio']:.3f}"
            )
        
        with col4:
            risk_return_ratio = portfolio_metrics['return'] / portfolio_metrics['volatility']
            st.metric(
                "Risk-Return Ratio",
                f"{risk_return_ratio:.3f}"
            )
        
        # Optimal weights visualization
        st.subheader("🥧 Optimal Asset Allocation")
        
        weights = result['optimal_weights']
        
        # Create pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=assets,
            values=list(weights.values()),
            hole=0.3,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig_pie.update_layout(
            title="Optimal Portfolio Allocation",
            height=500,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Weights table
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Asset Weights**")
            weights_df = pd.DataFrame([
                {'Asset': asset, 'Weight': f"{weight:.2%}", 'Amount': f"${weight*10000:.0f}"} 
                for asset, weight in zip(assets, weights.values())
            ])
            st.dataframe(weights_df, use_container_width=True)
        
        with col2:
            st.write("**Individual Asset Metrics**")
            individual_metrics = result['individual_metrics']
            
            metrics_data = []
            for symbol, asset in zip(symbols, assets):
                if symbol in individual_metrics:
                    metrics = individual_metrics[symbol]
                    metrics_data.append({
                        'Asset': asset,
                        'Weight': f"{metrics['weight']:.2%}",
                        'Expected Return': f"{metrics['return']*100:.2f}%",
                        'Volatility': f"{metrics['volatility']*100:.2f}%",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}"
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("🔗 Asset Correlation Matrix")
        
        correlation_matrix = result['correlation_matrix']
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=assets,
            y=assets,
            colorscale='RdYlBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size":10},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title="Asset Correlation Matrix",
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlation analysis
        avg_correlation = correlation_matrix.mean().mean()
        max_correlation = correlation_matrix.where(np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)).max().max()
        min_correlation = correlation_matrix.where(np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)).min().min()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Correlation", f"{avg_correlation:.3f}")
        
        with col2:
            st.metric("Maximum Correlation", f"{max_correlation:.3f}")
        
        with col3:
            st.metric("Minimum Correlation", f"{min_correlation:.3f}")
        
        # Risk Analysis
        st.subheader("⚠️ Risk Analysis")
        
        risk_analysis = optimizer.portfolio_risk_analysis(symbols, weights)
        
        if risk_analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Value at Risk (VaR)**")
                st.metric("VaR (95%)", f"{risk_analysis['var_95']*100:.2f}%")
                st.metric("VaR (99%)", f"{risk_analysis['var_99']*100:.2f}%")
            
            with col2:
                st.write("**Conditional VaR (CVaR)**")
                st.metric("CVaR (95%)", f"{risk_analysis['cvar_95']*100:.2f}%")
                st.metric("CVaR (99%)", f"{risk_analysis['cvar_99']*100:.2f}%")
            
            with col3:
                st.write("**Portfolio Risk Metrics**")
                st.metric("Maximum Drawdown", f"{risk_analysis['max_drawdown']*100:.2f}%")
                st.metric("Diversification Ratio", f"{risk_analysis['diversification_ratio']:.3f}")
        
        # Efficient Frontier
        st.subheader("📈 Efficient Frontier")
        
        with st.spinner("Generating efficient frontier..."):
            efficient_frontier = optimizer.generate_efficient_frontier(symbols, num_portfolios=5000)
        
        if efficient_frontier is not None and not efficient_frontier.empty:
            fig_frontier = go.Figure()
            
            # Scatter plot of portfolios
            fig_frontier.add_trace(go.Scatter(
                x=efficient_frontier['Volatility'],
                y=efficient_frontier['Return'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=efficient_frontier['Sharpe_Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Possible Portfolios',
                text=efficient_frontier['Sharpe_Ratio'].round(3),
                hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe: %{text}<extra></extra>'
            ))
            
            # Highlight optimal portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[portfolio_metrics['volatility']],
                y=[portfolio_metrics['return']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Optimal Portfolio'
            ))
            
            fig_frontier.update_layout(
                title="Efficient Frontier",
                xaxis_title="Volatility (Annual)",
                yaxis_title="Expected Return (Annual)",
                height=500
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Monte Carlo Simulation
        st.subheader("🎲 Monte Carlo Portfolio Simulation")
        
        simulation_days = st.selectbox("Simulation Period", [30, 90, 180, 365], index=2)
        num_simulations = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=1)
        
        if st.button("🎯 Run Monte Carlo Simulation"):
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = optimizer.monte_carlo_simulation(
                    symbols, weights, days=simulation_days, simulations=num_simulations
                )
            
            if mc_results:
                st.session_state.mc_results = mc_results
                st.session_state.simulation_days = simulation_days
        
        if 'mc_results' in st.session_state:
            mc_results = st.session_state.mc_results
            sim_days = st.session_state.simulation_days
            
            # Monte Carlo metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"Expected Value ({sim_days}d)",
                    f"${mc_results['mean_final_value']:.4f}"
                )
            
            with col2:
                st.metric(
                    "Probability of Loss",
                    f"{mc_results['probability_of_loss']*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Expected Return",
                    f"{mc_results['expected_return']*100:+.2f}%"
                )
            
            with col4:
                st.metric(
                    "Best Case Scenario",
                    f"${mc_results['best_case']:.4f}"
                )
            
            # Monte Carlo distribution
            fig_mc = go.Figure()
            
            fig_mc.add_trace(go.Histogram(
                x=mc_results['final_values'],
                nbinsx=50,
                name="Portfolio Value Distribution",
                opacity=0.7
            ))
            
            # Add percentile lines
            fig_mc.add_vline(
                x=mc_results['percentile_5'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"5th Percentile: ${mc_results['percentile_5']:.3f}"
            )
            
            fig_mc.add_vline(
                x=mc_results['percentile_95'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"95th Percentile: ${mc_results['percentile_95']:.3f}"
            )
            
            fig_mc.add_vline(
                x=mc_results['median_final_value'],
                line_dash="solid",
                line_color="blue",
                annotation_text=f"Median: ${mc_results['median_final_value']:.3f}"
            )
            
            fig_mc.update_layout(
                title=f"Monte Carlo Simulation Results ({sim_days} days, {num_simulations:,} simulations)",
                xaxis_title="Final Portfolio Value",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Risk scenarios
            st.write("**Risk Scenarios**")
            
            scenarios_df = pd.DataFrame([
                {"Scenario": "Best Case", "Value": f"${mc_results['best_case']:.4f}", "Return": f"{(mc_results['best_case']-1)*100:+.2f}%"},
                {"Scenario": "95th Percentile", "Value": f"${mc_results['percentile_95']:.4f}", "Return": f"{(mc_results['percentile_95']-1)*100:+.2f}%"},
                {"Scenario": "75th Percentile", "Value": f"${mc_results['percentile_75']:.4f}", "Return": f"{(mc_results['percentile_75']-1)*100:+.2f}%"},
                {"Scenario": "Median", "Value": f"${mc_results['median_final_value']:.4f}", "Return": f"{(mc_results['median_final_value']-1)*100:+.2f}%"},
                {"Scenario": "25th Percentile", "Value": f"${mc_results['percentile_25']:.4f}", "Return": f"{(mc_results['percentile_25']-1)*100:+.2f}%"},
                {"Scenario": "5th Percentile", "Value": f"${mc_results['percentile_5']:.4f}", "Return": f"{(mc_results['percentile_5']-1)*100:+.2f}%"},
                {"Scenario": "Worst Case", "Value": f"${mc_results['worst_case']:.4f}", "Return": f"{(mc_results['worst_case']-1)*100:+.2f}%"}
            ])
            
            st.dataframe(scenarios_df, use_container_width=True)
        
        # Rebalancing Strategy
        st.subheader("⚖️ Rebalancing Strategy")
        
        rebalance_threshold = st.slider(
            "Rebalancing Threshold (%)",
            1, 20, 5,
            help="Trigger rebalancing when asset weights deviate by this percentage"
        ) / 100
        
        if st.button("📊 Check Rebalancing Needs"):
            with st.spinner("Analyzing portfolio drift..."):
                rebalancing_analysis = optimizer.rebalancing_strategy(
                    symbols, weights, threshold=rebalance_threshold
                )
            
            if rebalancing_analysis:
                if rebalancing_analysis['needs_rebalancing']:
                    st.warning("⚠️ Portfolio rebalancing recommended!")
                    
                    rebalance_actions = rebalancing_analysis['rebalancing_actions']
                    
                    if rebalance_actions:
                        rebalance_df = pd.DataFrame([
                            {
                                'Asset': symbol.split('-')[0],
                                'Current Weight': f"{action['current_weight']:.2%}",
                                'Target Weight': f"{action['target_weight']:.2%}",
                                'Deviation': f"{action['deviation']:+.2%}",
                                'Action': action['action'],
                                'Amount': f"{action['rebalance_amount']:.2%}"
                            }
                            for symbol, action in rebalance_actions.items()
                        ])
                        
                        st.dataframe(rebalance_df, use_container_width=True)
                else:
                    st.success("✅ Portfolio is well balanced - no rebalancing needed")
                
                # Current vs target weights comparison
                current_weights = rebalancing_analysis['current_weights']
                target_weights = rebalancing_analysis['target_weights']
                
                fig_rebalance = go.Figure()
                
                asset_names = [s.split('-')[0] for s in symbols]
                
                fig_rebalance.add_trace(go.Bar(
                    name='Current Weights',
                    x=asset_names,
                    y=list(current_weights.values()),
                    marker_color='lightblue'
                ))
                
                fig_rebalance.add_trace(go.Bar(
                    name='Target Weights',
                    x=asset_names,
                    y=list(target_weights.values()),
                    marker_color='darkblue'
                ))
                
                fig_rebalance.update_layout(
                    title="Current vs Target Portfolio Weights",
                    xaxis_title="Assets",
                    yaxis_title="Weight",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_rebalance, use_container_width=True)
        
        # Portfolio Recommendations
        st.subheader("💡 Portfolio Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Optimization Insights**")
            
            # Analyze portfolio concentration
            max_weight = max(weights.values())
            num_significant_assets = sum(1 for w in weights.values() if w > 0.05)  # Assets with >5% allocation
            
            if max_weight > 0.6:
                st.warning("⚠️ Portfolio is highly concentrated in one asset")
            elif max_weight > 0.4:
                st.info("ℹ️ Portfolio has a dominant asset position")
            else:
                st.success("✅ Portfolio shows good diversification")
            
            st.write(f"• Number of significant positions (>5%): {num_significant_assets}")
            st.write(f"• Largest position: {max_weight:.1%}")
            st.write(f"• Average correlation: {avg_correlation:.3f}")
            
            # Risk assessment
            if portfolio_metrics['volatility'] > 0.5:
                st.error("🔴 High volatility portfolio - consider risk management")
            elif portfolio_metrics['volatility'] > 0.3:
                st.warning("🟡 Moderate volatility - suitable for balanced investors")
            else:
                st.success("🟢 Low volatility - conservative portfolio")
        
        with col2:
            st.write("**Investment Recommendations**")
            
            # Sharpe ratio analysis
            if portfolio_metrics['sharpe_ratio'] > 1.5:
                st.success("🟢 Excellent risk-adjusted returns")
            elif portfolio_metrics['sharpe_ratio'] > 1.0:
                st.info("ℹ️ Good risk-adjusted returns")
            elif portfolio_metrics['sharpe_ratio'] > 0.5:
                st.warning("🟡 Moderate risk-adjusted returns")
            else:
                st.error("🔴 Poor risk-adjusted returns")
            
            # Diversification recommendations
            if avg_correlation > 0.8:
                st.warning("⚠️ High correlation - consider adding uncorrelated assets")
            elif avg_correlation < 0.3:
                st.success("✅ Well diversified portfolio")
            
            # Expected return assessment
            if portfolio_metrics['return'] > 0.2:
                st.info("📈 High expected returns with corresponding risk")
            elif portfolio_metrics['return'] > 0.1:
                st.info("📊 Moderate expected returns")
            else:
                st.warning("📉 Conservative expected returns")
        
        # Export portfolio data
        st.subheader("💾 Export Portfolio Data")
        
        # Prepare export data
        portfolio_summary = {
            'optimization_method': optimization_method,
            'expected_return': portfolio_metrics['return'],
            'volatility': portfolio_metrics['volatility'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'risk_free_rate': risk_free_rate,
            'assets': assets,
            'symbols': symbols,
            'weights': list(weights.values())
        }
        
        portfolio_df = pd.DataFrame([
            {
                'Asset': asset,
                'Symbol': symbol,
                'Weight': weight,
                'Expected_Return': individual_metrics[symbol]['return'] if symbol in individual_metrics else 0,
                'Volatility': individual_metrics[symbol]['volatility'] if symbol in individual_metrics else 0,
                'Sharpe_Ratio': individual_metrics[symbol]['sharpe_ratio'] if symbol in individual_metrics else 0
            }
            for asset, symbol, weight in zip(assets, symbols, weights.values())
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = portfolio_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Portfolio CSV",
                data=csv_data,
                file_name=f"optimized_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            import json
            json_data = json.dumps(portfolio_summary, indent=2)
            st.download_button(
                label="📥 Download Portfolio JSON",
                data=json_data,
                file_name=f"optimized_portfolio_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        # Portfolio optimization guide
        st.info("👆 Select your cryptocurrency assets and click 'Optimize Portfolio' to begin")
        
        st.subheader("📚 Portfolio Optimization Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Modern Portfolio Theory**")
            st.write("• **Diversification**: Reduce risk through asset allocation")
            st.write("• **Efficient Frontier**: Maximize return for given risk level")
            st.write("• **Sharpe Ratio**: Risk-adjusted return optimization")
            st.write("• **Correlation**: Lower correlation = better diversification")
            
            st.write("**Risk Metrics**")
            st.write("• **VaR**: Potential loss at given confidence level")
            st.write("• **CVaR**: Expected loss beyond VaR threshold")
            st.write("• **Maximum Drawdown**: Largest peak-to-trough decline")
        
        with col2:
            st.write("**Optimization Methods**")
            st.write("• **Sharpe Ratio**: Maximizes risk-adjusted returns")
            st.write("• **Minimum Variance**: Minimizes portfolio volatility")
            
            st.write("**Best Practices**")
            st.write("• Select 3-10 assets for optimal diversification")
            st.write("• Use longer time periods for stable results")
            st.write("• Rebalance periodically to maintain target weights")
            st.write("• Consider correlation when selecting assets")
            st.write("• Monitor risk metrics regularly")

if __name__ == "__main__":
    main()
