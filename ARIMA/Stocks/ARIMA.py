import pandas as pd
import time
import csv
import os
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
import numpy as np
from preprocessor import Preprocessor
import warnings
import preprocessor
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

def getStockName(file):
    # In format "preprocessed_{StockName}.csv"
    parts = file.split("_")   # split the string by "_" characters
    ticker = parts[1].split(".")[0]   # extract the second part and remove the ".csv" extension
    return ticker

def RUN_ARIMA(file, window_size):
    # Where files will be stored
    stock_name = getStockName(f"{file}")
    folder_name = f"{stock_name}_Folder"
    file_name = f"{folder_name}/{window_size}_{stock_name}.csv"   
    
    # Load your data and preprocess it
    data = pd.read_csv(file)
    
    # Remove first row, training data / prediction includes the day of
    data = data.iloc[1 + (preprocessor.max_size-window_size):]
    data.set_index('Date', inplace=True)

    # Function to train and predict using the ARIMA model
    def train_and_predict(train_data):
        # Create the time series data for ARIMA
        ts_data = pd.Series(train_data['Close'].values, index=pd.to_datetime(train_data.index))  # Closing price
    
        # Find the best ARIMA model
        best_model = auto_arima(
            ts_data,
            seasonal=True,
            m = 5,
            stepwise=True,
            suppress_warnings=True,
            max_order=None,
            trace=False,
            error_action='ignore',
        )
    
        # Forecast the next day
        forecast = best_model.predict(n_periods=1)
                
        return forecast.iloc[0]
    
    def calculate_daily_gain_loss(file_name):
        new_data = pd.read_csv(file_name)
        changes_history= []
        balance_history = []
        balance = 0
        for i in range(len(new_data) - 1):
            current_day = new_data.iloc[i]
            next_day = new_data.iloc[i + 1]
            
            if str(current_day["Higher"]) == str(current_day["Buy"]):
                change = abs(next_day["Close"] - current_day["Close"])
            else:
                change = -abs((next_day["Close"] - current_day["Close"]))
            
            changes_history.append(change)
            balance += change
            balance_history.append(balance)
    
        return changes_history, balance_history
    
    # Compare the predicted close price of the next day with the current day
    def is_next_day_higher(current_day, next_day):
        if next_day > current_day:
            return "True"
        return "False"
    
    try:
        # Moving window loop
        predictions = []
        
        # Plus 1 to include the data of that day
        for i in range(len(data) - window_size):
            train_data = data.iloc[i : i + window_size]
            prediction = train_and_predict(train_data)
            predictions.append(prediction)
        
        
        # Append results to the new DataFrame and save it as a CSV file
        new_data = data.iloc[window_size - 1:].copy()
        
        results = [is_next_day_higher(new_data["Close"].values[i], predictions[i]) for i in range(len(predictions))]
    
        results.append("NONE")
    
        new_data['Higher'] = results
        new_data.to_csv(file_name)
        
        time.sleep(0.5)
        
        changes_history, balance_history = calculate_daily_gain_loss(file_name)
        
        plot_balance_history(folder_name, window_size, balance_history, stock_name)
        saveDifferences(folder_name, window_size, changes_history, stock_name)
        
        print(f"Done {window_size}")
        return(True)
    except:
        return(False)
    

    
    
def plot_balance_history(folder_name, size, balance_history, stock):
    # Create a list of time points
    time_points = list(range(len(balance_history)))

    # Create a smoothed line curve of the balance over time
    x_smooth = np.linspace(time_points[0], time_points[-1], 200)
    y_smooth = np.interp(x_smooth, time_points, balance_history)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=2)

    # Add the last balance on the graph
    year3_bal = balance_history[-1]
    plt.text(0.95, 0.01, f"End Balance: {year3_bal:.2f}", transform=plt.gcf().transFigure, fontsize=10, ha='right', va='bottom')

    # Add a black dotted line horizontally from 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(axis='both', width=0.5, labelsize=12)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.set_title(f'{stock} Price over Time, size: {size}', fontsize=16)

    # Save the plot in the specified folder with high quality
    file_path = os.path.join(folder_name, f'stock_price_{size}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    
def saveDifferences(folder_name, size, difference_history, stock):
    # Write the balance values to the file
    with open(f'{folder_name}/difference_{stock}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        row = [size] + difference_history
        writer.writerow(row)
        
def saveBalances(folder_name, size, history, stock):
    n = len(history)
    middle_index = n // 2   # integer division to get the index of the middle element
    first_half = history[:middle_index]   # Get the first half of the list
    second_half = history[middle_index:]   # Get the second half of the list

    # Check if the file exists, create it if not
    if not os.path.isfile(f'{folder_name}/balance_{stock}.csv'):
        with open(f'{folder_name}/balance_{stock}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Size', 'Year 1', 'Year 2'])

    # Write the balance values to the file
    with open(f'{folder_name}/balance_{stock}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([size, first_half, second_half])

def run_concurrent(files):
    for file in files:
        run_single_file(file)
        
def run_single_file(file):
    Preprocessor(file)
    # create the new directory if it doesn't exist
    folder = getStockName(f"preprocessed_{file}") + "_Folder"

    time.sleep(1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    sizes = range(21, 121, 1)
    
    # create a list to store the futures of the concurrent tasks
    futures = []
    
    go = '''
    for size in sizes:
        RUN_ARIMA(f"preprocessed_{file}", size)
    
    import sys
    sys.exit()'''

    # determine the maximum number of threads based on the number of available cores
    max_threads = multiprocessing.cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for size in sizes:
            # submit the task to the executor and store the future object
            future = executor.submit(RUN_ARIMA, f"preprocessed_{file}", size)
            futures.append(future)
        
        # use tqdm to create a progress bar for the futures list
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            # check if the future returned an exception and re-raise it if it did
            if future.exception() is not None:
                raise future.exception()
   

if __name__ == '__main__':    
    multiprocessing.freeze_support()
    files = ["AAPL.csv", "AMZN.csv", "GOOGL.csv", 
             "JNJ.csv", "KO.csv", "META.csv", 
             "MSFT.csv", "PG.csv", "V.csv", "WMT.csv"]
    
    files = ["META.csv", 
             "MSFT.csv", "PG.csv", "V.csv", "WMT.csv"]

    # Run the code
    run_concurrent(files)


            
            
            
            
            
            
            