import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
"SPY gives big gains"
# Load data
df = pd.read_csv("TSLA_stock_data.csv", parse_dates=[0]) # Chose your stock using a raw data CSV file.
ticker = "TSLA"

# Clean and prepare data
data = df.drop(index=0).dropna()
data['Close'] = data['Close'].astype(float)
data['High'] = data['High'].astype(float)
data['Low'] = data['Low'].astype(float)
data['Open'] = data['Open'].astype(float)
data['Volume'] = data['Volume'].astype(float)
data = data.rename(columns={'Price': 'Date'})
data['Date'] = pd.to_datetime(data['Date'])

# Calculate RSI
def calculate_rsi(data, period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

# RSI Strategy
def rsi_strategy(data):
    """Apply an RSI-based trading strategy."""
    data['Signal'] = 0  # 0 = No Signal, 1 = Buy, -1 = Sell

    # Buy when RSI crosses below 30, Sell when RSI crosses above 70
    data.loc[data['RSI'] < 30, 'Signal'] = 1  # Buy Signal

    return data

# Backtest Strategy
def backtest_strategy(data, initial_balance, take_profit=1.05, stop_loss_multiplier=0.99):
    """Simulate the profitability of the RSI strategy (Buy, Take Profit, Stop Loss, Short)."""
    long_balance = initial_balance  # Starting cash for long positions
    shares_held_long = 0  # Number of shares owned (long)
    portfolio_value = []  # Track portfolio value over time
    purchase_prices = []  # Track purchase prices for long positions

    for i, row in data.iterrows():

        # Long Logic
        if row['Signal'] == 1 and long_balance > 0:  # Buy signal
            shares_bought = long_balance // row['Close']
            if shares_bought > 0:
                long_balance -= shares_bought * row['Close']
                shares_held_long += shares_bought
                purchase_prices.append((row['Close'], shares_bought))

        # Long Take Profit
        shares_to_sell_tp = 0
        for j in range(len(purchase_prices)):
            price, quantity = purchase_prices[j]
            if row['Close'] >= price * take_profit:
                shares_to_sell_tp += quantity
                long_balance += quantity * row['Close']
                purchase_prices[j] = (price, 0)

        purchase_prices = [(p,q) for p,q in purchase_prices if q>0]
        shares_held_long -= shares_to_sell_tp

        # Long Stop Loss
        shares_to_sell_sl = 0
        for j in range(len(purchase_prices)):
            price, quantity = purchase_prices[j]
            if row['Close'] < price * stop_loss_multiplier:
                shares_to_sell_sl += quantity
                long_balance += quantity * row['Close']
                purchase_prices[j] = (price, 0)

        purchase_prices = [(p,q) for p,q in purchase_prices if q>0]

        total_value = long_balance + (shares_held_long * row['Close'])  # Portfolio value calculation
        portfolio_value.append(total_value)

    data['Portfolio_Value'] = portfolio_value
    return data

# Calculate RSI
data = calculate_rsi(data)

# Apply RSI trading strategy
data = rsi_strategy(data)

# Backtest the strategy
data = backtest_strategy(data, initial_balance=1000)

data = data.set_index('Date')

# Load risk-free rate data
risk_free_df = pd.read_csv("2015_risk_free.csv", parse_dates=[0])
risk_free_df = risk_free_df.rename(columns={'observation_date': 'Date', 'DGS10': 'Risk_Free_Rate'})
risk_free_df['Date'] = pd.to_datetime(risk_free_df['Date'])

# Merge risk-free rate with strategy data
data = data.merge(risk_free_df, on='Date', how='left')
data['Risk_Free_Rate'].ffill()  # Forward fill missing risk-free rates

# Calculate daily returns
data['Daily_Returns'] = data['Portfolio_Value'].pct_change()

# Calculate excess returns (Strategy Return - Risk-Free Rate)
data['Excess_Returns'] = data['Daily_Returns'] - (data['Risk_Free_Rate'] / 252)  # Convert annual rate to daily


# Calculate Sharpe Ratio
sharpe_ratio = data['Excess_Returns'].mean() / data['Excess_Returns'].std()
sharpe_ratio *= np.sqrt(252)  # Annualize the Sharpe Ratio

# Visualize the strategy and portfolio value
def plot_strategy_and_portfolio(data):
    """Plot the strategy and portfolio performance."""
    plt.figure(figsize=(14, 10))

    # Plot stock price with buy signals
    plt.subplot(4, 1, 1)
    plt.plot(data['Close'], label='Close Price', alpha=0.7)
    plt.scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], label='Buy Signal', marker='^', color='green')
    plt.title(f'{ticker} Price with RSI Strategy')
    plt.legend()

    # Plot RSI
    plt.subplot(4, 1, 2)
    plt.plot(data['RSI'], label='RSI', color='orange', alpha=0.7)
    plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
    plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
    plt.title('RSI Indicator')
    plt.legend()

    # Plot portfolio value
    plt.subplot(4, 1, 3)
    plt.plot(data['Portfolio_Value'], label='Portfolio Value', color='blue', alpha=0.7)
    plt.title('Portfolio Value Over Time')
    plt.legend()

    # Plot excess returns
    plt.subplot(4,1,4)
    plt.plot(data['Excess_Returns'], label='Excess Returns', color='green', alpha=0.7)
    plt.title('Portfolio Value and Excess Returns Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_strategy_and_portfolio(data)

# Print final portfolio value
final_value = data['Portfolio_Value'].iloc[-1]
print(f"Final Portfolio Value: ${final_value:.2f}")

print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
