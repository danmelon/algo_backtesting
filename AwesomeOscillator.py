import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
"using AAPL shows decay (SPY does nothing then surges)"
# Load data
df = pd.read_csv("VIX_stock_data.csv", parse_dates=[0])
ticker = "vix"

# Clean and prepare data
data = df.drop(index=0).dropna()
data['Close'] = data['Close'].astype(float)
data['High'] = data['High'].astype(float)
data['Low'] = data['Low'].astype(float)
data['Open'] = data['Open'].astype(float)
data['Volume'] = data['Volume'].astype(float)
data = data.rename(columns={'Price': 'Date'})
data['Date'] = pd.to_datetime(data['Date'])


# Calculate SMA (34-day SMA of median prices which is subtracted from the 5 day SMA of median prices)

# AO Strategy
def calculate_ao(data, fast_length = 1, slow_length = 50):
    
    data['Median Price'] = (data['High'] + data['Low']) / 2
    data['Fast SMA'] = data['Median Price'].rolling(window=fast_length).mean()
    data['Slow SMA'] = data['Median Price'].rolling(window=slow_length).mean()
    data['AO'] = data['Fast SMA'] - data['Slow SMA']

    return data

def ao_strategy(data):
    """Apply an AO-based trading strategy."""
    data['Signal'] = 0  # 0 = No Signal, 1 = Buy, -1 = Sell

    cross_above = (data['AO'] > 0) & (data['AO'].shift(1) < 0)
    cross_below = (data['AO'] < 0) & (data['AO'].shift(1) > 0)

    data.loc[cross_above, 'Signal'] = 1
    data.loc[cross_below, 'Signal'] = -1

    return data

# Backtest Strategy
def backtest_strategy(data, initial_balance, take_profit=1.075):
    """Simulate the profitability of the RSI strategy."""
    balance = initial_balance  # Starting cash
    shares_held = 0  # Number of shares owned
    portfolio_value = []  # Track portfolio value over time
    purchase_price = None  # Track the price at which shares were bought

    for i, row in data.iterrows():
        one_week_low = data['Close'].iloc[max(0, i - 7):i].min() # change the i-7 part to change number of day low it uses

        if row['Signal'] == 1 and balance > 0:  # Buy signal
            shares_bought = balance // row['Close']  # Integer shares
            balance -= shares_bought * row['Close']
            shares_held += shares_bought
            purchase_price = row['Close']  # Track purchase price

        ''' This is redundant since we are using the take_profit and not implimenting a shorting strategy
        elif row['Signal'] == -1 and shares_held > 0:  # Sell signal
            balance += shares_held * row['Close']
            shares_held = 0
            purchase_price = None  # Reset purchase price
        '''
        # Take Profit Logic
        if shares_held > 0 and row['Close'] >= purchase_price * take_profit:
            balance += shares_held * row['Close']
            shares_held = 0
            purchase_price = None

        # Stop Loss Logic
        if shares_held > 0 and row['Close'] < one_week_low:
            balance += shares_held * row['Close']
            shares_held = 0
            purchase_price = None

        # Calculate portfolio value: cash + (shares held * current price)
        total_value = balance + (shares_held * row['Close'])
        portfolio_value.append(total_value)

    # Add portfolio value to the DataFrame
    data['Portfolio_Value'] = portfolio_value
    return data

# Calculate AO
data = calculate_ao(data)

# Apply AO trading strategy
data = ao_strategy(data)

# Backtest the strategy
data = backtest_strategy(data, initial_balance=10000)

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

# Calculate Maximum Drawdown
data['Cumulative_Max'] = data['Portfolio_Value'].cummax()
data['Drawdown'] = (data['Portfolio_Value'] - data['Cumulative_Max']) / data['Cumulative_Max']
max_drawdown = data['Drawdown'].min()

# Visualize the strategy and portfolio value
def plot_strategy_and_portfolio(data):
    """Plot the strategy and portfolio performance."""
    plt.figure(figsize=(14, 10))

    # Plot stock price with buy/sell signals
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price', alpha=0.7)
    plt.scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], label='Buy Signal', marker='^', color='green')
    plt.scatter(data.index[data['Signal'] == -1], data['Close'][data['Signal'] == -1], label='Sell Signal', marker='v', color='red')
    plt.title(f'{ticker} Price with RSI Strategy')
    plt.legend()

    #plot zero line
    plt.axhline(0, color='gray', linestyle='--')

    # Plot portfolio value
    plt.subplot(3, 1, 2)
    plt.plot(data['Portfolio_Value'], label='Portfolio Value', color='blue', alpha=0.7)
    plt.title('Portfolio Value Over Time')
    plt.legend()

    # Plot excess returns
    plt.subplot(3, 1, 3)
    plt.plot(data['Excess_Returns'], label='Excess Returns', color='green', alpha=0.7)
    plt.title('Portfolio Value and Excess Returns Over Time')
    plt.legend()

    # Plot Drawdown
    #plt.subplot(4, 1, 4) # Added new subplot for drawdown
    plt.plot(data['Drawdown'], label='Drawdown', color='orange', alpha=0.7)
    plt.title('Drawdown Over Time')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')


    plt.tight_layout()
    plt.show()

plot_strategy_and_portfolio(data)

# Print final portfolio value
final_value = data['Portfolio_Value'].iloc[-1]
print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")

# Calculate Annualised Returns
initial_balance = 10000

annualised_return = 100*(((final_value/initial_balance)**(1/((data.shape[0])/252))) -1)
print(f"Annualised return: {annualised_return:.2f}%")
