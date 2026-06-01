import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("Libraries Imported Successfully")
trades = pd.read_csv("historical_data.csv")

print("Shape:", trades.shape)
print("\nColumns:")
print(trades.columns)

trades.head()
sentiment = pd.read_csv("fear_greed_index.csv")

print("Shape:", sentiment.shape)
print("\nColumns:")
print(sentiment.columns)

sentiment.head()
print("Trader Dataset Missing Values:")
print(trades.isnull().sum())

print("\nSentiment Dataset Missing Values:")
print(sentiment.isnull().sum())
trades['date'] = pd.to_datetime(
    trades['Timestamp IST'],
    dayfirst=True
).dt.date

sentiment['date'] = pd.to_datetime(
    sentiment['date']
).dt.date

print("Date Conversion Completed")
merged = pd.merge(
    trades,
    sentiment[['date', 'classification', 'value']],
    on='date',
    how='left'
)

print("Merged Shape:", merged.shape)

merged.head()
print(
    merged['classification']
    .value_counts()
)
total_profit = merged.groupby(
    'classification'
)['Closed PnL'].sum()

print(total_profit)
avg_profit = merged.groupby(
    'classification'
)['Closed PnL'].mean()

print(avg_profit)
trade_count = merged.groupby(
    'classification'
).size()

print(trade_count)
volume = merged.groupby(
    'classification'
)['Size USD'].sum()

print(volume)
plt.figure(figsize=(8,5))

total_profit.plot(kind='bar')

plt.title("Total Profit by Sentiment")
plt.ylabel("Profit")
plt.show()
plt.figure(figsize=(8,5))

avg_profit.plot(kind='bar')

plt.title("Average Profit by Sentiment")
plt.ylabel("Average PnL")

plt.show()
merged = pd.merge(
    trades,
    sentiment[['date', 'classification', 'value']],
    on='date',
    how='left'
)
buy_sell = merged.groupby(
    ['classification','Side']
)['Closed PnL'].mean()

print(buy_sell)
summary = merged.groupby(
    'classification'
).agg({
    'Closed PnL':['sum','mean'],
    'Size USD':'sum'
})

summary.to_csv(
    'sentiment_analysis_results.csv'
)

print("Results Saved Successfully")
merged['Win'] = merged['Closed PnL'] > 0

win_rate = merged.groupby('classification')['Win'].mean() * 100

print(win_rate)
correlation = merged[
    ['Closed PnL','value']
].corr()

print(correlation)





