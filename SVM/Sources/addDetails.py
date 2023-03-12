import pandas as pd
import numpy as np


# All the files`
#file = 'AAPL.csv'
#file = 'TSLA.csv'
#file = 'JNJ.csv'

# Load the CSV file into a pandas dataframe
df = pd.read_csv(file, header=0)

# Drop the first row, which contains the header
# Define the periods for the MACD calculation
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

# Calculate the MACD and signal line
macd = ema_12 - ema_26
signal = macd.ewm(span=9, adjust=False).mean()

# Calculate the histogram
hist = macd - signal

# Define the periods and standard deviation for the Bollinger Bands calculation
ma = df['Close'].rolling(window=20).mean()
std = df['Close'].rolling(window=20).std()

# Calculate the upper and lower Bollinger Bands
upper_band = ma + 2 * std
lower_band = ma - 2 * std

# Add the MACD, signal line, histogram, upper and lower Bollinger Bands as new columns in the dataframe
df['MACD'] = macd
df['Signal Line'] = signal
df['Histogram'] = hist
df['Upper Band'] = upper_band
df['Lower Band'] = lower_band

# Add a Buy column based on the close price comparison
df['Buy'] = np.where(df['Close'] < df['Close'].shift(-1), True, False)

# Save the updated dataframe back to the CSV file
df.to_csv(f'new{file}', index=False)
