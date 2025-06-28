import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.trading_days = 252
    
    def get_portfolio_data(self, symbols, period='1y'):
        """Get historical data for portfolio symbols"""
        try:
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
            
            if data:
                return pd.DataFrame(data).dropna()
            return None
            
        except Exception as e:
            print(f"Portfolio data fetch error: {str(e)}")
            return None
    
    def calculate_returns(self, prices):
        """Calculate daily returns"""
        try:
            returns = prices.pct_change().dropna()
            return returns
        except Exception as e:
            print(f"Returns calculation error: {str(e)}")
            return None
    
    def calculate_portfolio_metrics(self, weights, returns):
        """Calculate portfolio metrics"""
        try:
            # Portfolio return
            portfolio_return = np.sum(returns.mean() * weights) * self.trading_days
            
            # Portfolio volatility
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * self.trading_days, weights)))
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        except Exception as e:
            print(f"Portfolio metrics calculation error: {str(e)}")
            return None
    
    def negative_sharpe_ratio(self, weights, returns):
        """Objective function for optimization (negative Sharpe ratio)"""
        try:
            metrics = self.calculate_portfolio_metrics(weights, returns)
            if metrics:
                return -metrics['sharpe_ratio']
            return 1000  # Large penalty for invalid portfolios
        except:
            return 1000
    
    def optimize_portfolio(self, symbols, optimization_method='sharpe'):
        """Optimize portfolio using different methods"""
        try:
            # Get price data
            prices = self.get_portfolio_data(symbols)
            if prices is None or prices.empty:
                return None
            
            # Calculate returns
            returns = self.calculate_returns(prices)
            if returns is None or returns.empty:
                return None
            
            n_assets = len(symbols)
            
            # Constraints and bounds
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimize based on method
            if optimization_method == 'sharpe':
                result = minimize(
                    self.negative_sharpe_ratio,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    args=(returns,)
                )
            elif optimization_method == 'min_variance':
                result = minimize(
                    self.portfolio_variance,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    args=(returns,)
                )
            else:
                return None
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate metrics for optimal portfolio
                metrics = self.calculate_portfolio_metrics(optimal_weights, returns)
                
                # Calculate individual asset metrics
                individual_metrics = {}
                for i, symbol in enumerate(symbols):
                    asset_return = returns[symbol].mean() * self.trading_days
                    asset_volatility = returns[symbol].std() * np.sqrt(self.trading_days)
                    asset_sharpe = (asset_return - self.risk_free_rate) / asset_volatility
                    
                    individual_metrics[symbol] = {
                        'weight': optimal_weights[i],
                        'return': asset_return,
                        'volatility': asset_volatility,
                        'sharpe_ratio': asset_sharpe
                    }
                
                return {
                    'optimal_weights': dict(zip(symbols, optimal_weights)),
                    'portfolio_metrics': metrics,
                    'individual_metrics': individual_metrics,
                    'correlation_matrix': returns.corr(),
                    'optimization_success': True
                }
            else:
                return None
                
        except Exception as e:
            print(f"Portfolio optimization error: {str(e)}")
            return None
    
    def portfolio_variance(self, weights, returns):
        """Calculate portfolio variance"""
        try:
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * self.trading_days, weights)))
            return portfolio_vol ** 2
        except:
            return 1000
    
    def generate_efficient_frontier(self, symbols, num_portfolios=10000):
        """Generate efficient frontier"""
        try:
            prices = self.get_portfolio_data(symbols)
            if prices is None:
                return None
            
            returns = self.calculate_returns(prices)
            if returns is None:
                return None
            
            n_assets = len(symbols)
            results = np.zeros((3, num_portfolios))
            
            # Generate random weights
            np.random.seed(42)
            for i in range(num_portfolios):
                weights = np.random.random(n_assets)
                weights /= np.sum(weights)
                
                metrics = self.calculate_portfolio_metrics(weights, returns)
                if metrics:
                    results[0, i] = metrics['return']
                    results[1, i] = metrics['volatility']
                    results[2, i] = metrics['sharpe_ratio']
            
            # Create DataFrame
            efficient_frontier = pd.DataFrame({
                'Return': results[0],
                'Volatility': results[1],
                'Sharpe_Ratio': results[2]
            })
            
            return efficient_frontier
            
        except Exception as e:
            print(f"Efficient frontier generation error: {str(e)}")
            return None
    
    def calculate_var(self, returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        try:
            if isinstance(returns, pd.Series):
                sorted_returns = returns.sort_values()
                index = int(confidence_level * len(sorted_returns))
                var = sorted_returns.iloc[index]
                return abs(var)
            return None
        except Exception as e:
            print(f"VaR calculation error: {str(e)}")
            return None
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            if isinstance(returns, pd.Series):
                var = self.calculate_var(returns, confidence_level)
                cvar = returns[returns <= -var].mean()
                return abs(cvar)
            return None
        except Exception as e:
            print(f"CVaR calculation error: {str(e)}")
            return None
    
    def calculate_maximum_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            if isinstance(prices, pd.Series):
                cumulative = (1 + prices.pct_change()).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                return abs(max_drawdown)
            return None
        except Exception as e:
            print(f"Maximum drawdown calculation error: {str(e)}")
            return None
    
    def portfolio_risk_analysis(self, symbols, weights=None):
        """Comprehensive portfolio risk analysis"""
        try:
            prices = self.get_portfolio_data(symbols)
            if prices is None:
                return None
            
            returns = self.calculate_returns(prices)
            if returns is None:
                return None
            
            # Use equal weights if not provided
            if weights is None:
                weights = np.array([1/len(symbols)] * len(symbols))
            else:
                weights = np.array(list(weights.values()) if isinstance(weights, dict) else weights)
            
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Risk metrics
            var_95 = self.calculate_var(portfolio_returns, 0.05)
            var_99 = self.calculate_var(portfolio_returns, 0.01)
            cvar_95 = self.calculate_cvar(portfolio_returns, 0.05)
            cvar_99 = self.calculate_cvar(portfolio_returns, 0.01)
            
            # Portfolio prices for drawdown calculation
            portfolio_prices = (prices * weights).sum(axis=1)
            max_drawdown = self.calculate_maximum_drawdown(portfolio_prices)
            
            # Correlation analysis
            correlation_matrix = returns.corr()
            avg_correlation = correlation_matrix.mean().mean()
            
            # Diversification ratio
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * self.trading_days, weights)))
            weighted_avg_vol = np.sum(weights * returns.std() * np.sqrt(self.trading_days))
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'portfolio_volatility': portfolio_vol,
                'diversification_ratio': diversification_ratio,
                'average_correlation': avg_correlation,
                'correlation_matrix': correlation_matrix
            }
            
        except Exception as e:
            print(f"Portfolio risk analysis error: {str(e)}")
            return None
    
    def rebalancing_strategy(self, symbols, target_weights, threshold=0.05):
        """Generate rebalancing recommendations"""
        try:
            prices = self.get_portfolio_data(symbols, period='1mo')
            if prices is None:
                return None
            
            # Calculate current weights (assuming equal initial investment)
            current_prices = prices.iloc[-1]
            initial_prices = prices.iloc[0]
            
            # Calculate returns for each asset
            asset_returns = (current_prices / initial_prices) - 1
            
            # Calculate current weights based on performance
            initial_values = np.array([1/len(symbols)] * len(symbols))
            current_values = initial_values * (1 + asset_returns)
            current_weights = current_values / current_values.sum()
            
            # Target weights
            target_weights_array = np.array(list(target_weights.values()))
            
            # Calculate deviations
            deviations = current_weights - target_weights_array
            
            # Check if rebalancing is needed
            needs_rebalancing = np.any(np.abs(deviations) > threshold)
            
            rebalancing_actions = {}
            for i, symbol in enumerate(symbols):
                deviation = deviations[i]
                if abs(deviation) > threshold:
                    action = "SELL" if deviation > 0 else "BUY"
                    amount = abs(deviation)
                    rebalancing_actions[symbol] = {
                        'action': action,
                        'current_weight': current_weights[i],
                        'target_weight': target_weights_array[i],
                        'deviation': deviation,
                        'rebalance_amount': amount
                    }
            
            return {
                'needs_rebalancing': needs_rebalancing,
                'rebalancing_actions': rebalancing_actions,
                'current_weights': dict(zip(symbols, current_weights)),
                'target_weights': target_weights,
                'threshold': threshold
            }
            
        except Exception as e:
            print(f"Rebalancing strategy error: {str(e)}")
            return None
    
    def monte_carlo_simulation(self, symbols, weights, days=252, simulations=1000):
        """Monte Carlo simulation for portfolio projections"""
        try:
            prices = self.get_portfolio_data(symbols)
            if prices is None:
                return None
            
            returns = self.calculate_returns(prices)
            if returns is None:
                return None
            
            # Portfolio historical metrics
            portfolio_returns = (returns * list(weights.values())).sum(axis=1)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Monte Carlo simulation
            simulated_returns = np.random.normal(mean_return, std_return, (simulations, days))
            simulated_prices = np.zeros((simulations, days + 1))
            simulated_prices[:, 0] = 1  # Starting value of 1
            
            for sim in range(simulations):
                for day in range(1, days + 1):
                    simulated_prices[sim, day] = simulated_prices[sim, day - 1] * (1 + simulated_returns[sim, day - 1])
            
            # Calculate statistics
            final_values = simulated_prices[:, -1]
            
            return {
                'simulated_prices': simulated_prices,
                'final_values': final_values,
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'percentile_5': np.percentile(final_values, 5),
                'percentile_25': np.percentile(final_values, 25),
                'percentile_75': np.percentile(final_values, 75),
                'percentile_95': np.percentile(final_values, 95),
                'probability_of_loss': np.sum(final_values < 1) / simulations,
                'expected_return': np.mean(final_values) - 1,
                'worst_case': np.min(final_values),
                'best_case': np.max(final_values)
            }
            
        except Exception as e:
            print(f"Monte Carlo simulation error: {str(e)}")
            return None
