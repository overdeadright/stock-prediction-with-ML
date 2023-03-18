import numpy as np
import pandas as pd
import time 

max_size = 100 + 20

# How many rows will be missing values as NaN or 0 to calculate RSI and such. 
# Keep it during calculated, and drop at the end
learning_size = 21

# Stocks and Crypto historical data are from Yahoo
def Preprocessor(file, start_date = "2020-01-02", end_date = "2022-12-30"):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file, header=0)
    
    # Check if the start date is in the DataFrame
    if start_date not in df['Date'].values:
        print(start_date, df['Date'].values)
        raise ValueError('Start date is not in the DataFrame.')
        
    # Check if the end date is in the DataFrame
    if end_date not in df['Date'].values:
        print(end_date, df['Date'].values)
        raise ValueError('End date is not in the DataFrame.')
        
    # Get the row index for the specified date
    start_date_index = df[df['Date'] == start_date].index[0]

    # Moving average 200 requires at least 199 rows prior to it
    # Check if the date index is at least 199 + 100 for size analysis
    if start_date_index < max_size:
        raise ValueError(f"The specified date is not at least {max_size + learning_size} rows below the header. It is {start_date_index}.")

    # Drop rows 199 rows below the date index (max_size left for size)
    drop_index = start_date_index - learning_size - max_size 
    df.drop(df.index[:drop_index], inplace=True)
    
    # Sort the dataframe by the Date column in ascending order
    df.sort_values(by=['Date'], inplace=True)
    
    # Reset the index of the sorted dataframe
    df.reset_index(drop=True, inplace=True)

    # Define the periods for the MACD calculation
    # Use exponential moving averages with periods of 12 and 26 to calculate the MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD and signal line
    # Subtract the 26-period EMA from the 12-period EMA to calculate the MACD
    # Use a 9-period EMA of the MACD to calculate the signal line
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    # Calculate the Relative Strength Index (RSI)
    # Use a 14-period window to calculate the RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate the Average True Range (ATR)
    # Use a 14-period window to calculate the ATR
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    atr = true_range.rolling(window=14).mean()

    # Calculate the On-Balance Volume (OBV)
    # Use the previous day's closing price to calculate the OBV
    obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume']).cumsum()

    # Calculate the Stochastic Oscillator
    # Use a 14-period window to calculate the Stochastic Oscillator
    k_period = 14
    min_low = df['Low'].rolling(window=k_period).min()
    max_high = df['High'].rolling(window=k_period).max()
    stoch = 100 * (df['Close'] - min_low) / (max_high - min_low)

    # Calculate the Chaikin Money Flow (CMF)
    # Use a 20-period window to calculate the CMF
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mfv * df['Volume']
    cmf = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # Remove any rows with missing values from the dataframe
    df.dropna(inplace=True)
    
    # Add the new features as new columns in the dataframe
    df['MACD'] = macd
    df['Signal Line'] = signal
    df['RSI'] = rsi
    df['ATR'] = atr
    df['OBV'] = obv
    df['CMF'] = cmf
    
    # Add a Buy column based on the close price comparison
    df['Buy'] = np.where(df['Close'] < df['Close'].shift(-1), True, False)
    
    # Drop the 'Adj Close' column from the dataframe
    df.drop('Adj Close', axis=1, inplace=True)
    
    # Drop learning size
    drop_index = learning_size
    df.drop(df.index[:drop_index], inplace=True)
        
    # Up until end_date
    end_date_index = df[df['Date'] == end_date].index[0]
    df = df.iloc[:end_date_index+1]
    
    # Save the updated dataframe back to the CSV file
    # Start date should be index 101
    df.to_csv(f'preprocessed_{file}', index=False)
    
    # Let file load
    time.sleep(1)
        

