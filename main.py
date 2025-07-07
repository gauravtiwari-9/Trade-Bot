import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# data_fetcher
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data


# feature_engineering
def add_technical_indicators(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Signal'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # Buy if next day is higher
    df.dropna(inplace=True)
    return df


# strategy
def train_model(df):
    features = ['MA10', 'MA50', 'Return']
    X = df[features]
    y = df['Signal']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    df['Prediction'] = model.predict(X)
    return df, model


# simulator
def simulate_trades(df, initial_cash=10000):
    cash = initial_cash
    position = 0.0
    for i in range(len(df)):
        close_price = float(df['Close'].values[i])
        pred = int(df['Prediction'].values[i])
        if pred == 1 and cash >= close_price:
            position = cash / close_price
            cash = 0.0
        elif pred == 0 and position > 0:
            cash = position * close_price
            position = 0.0
    final_value = cash + position * float(df['Close'].values[-1])
    return final_value

# visualize
def plot_signals(df):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close Price')
    
    buy_signals = df[df['Prediction'] == 1]
    sell_signals = df[df['Prediction'] == 0]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='g')
    plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='r')
    
    plt.xlabel("Date")                # Label for x-axis
    plt.ylabel("Price (INR)")        # Label for y-axis
    plt.title("Trading Signals")
    plt.legend()
    plt.grid(True)                   # Enable grid lines
    
    # Save the plot instead of showing it
    plt.savefig("trade_signals.png", dpi=150)
    plt.close()


# main
def main():
    ticker = "SBIN.NS"
    start_date = "2025-01-01"
    end_date = "2025-06-10"
    initial_cash = 1000
    df = fetch_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        print("Error: No data fetched for the given ticker and date range.")
        return
    
    df = add_technical_indicators(df)
    
    if df.empty:
        print("Error: No data available after adding technical indicators.")
        return
    
    df, model = train_model(df)
    
    final_value = simulate_trades(df, initial_cash)
    print(f"Final Portfolio Value: ${final_value:.2f}")
    
    plot_signals(df)

if __name__ == "__main__":
    main()
