import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

def add_technical_indicators(file):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(file, header=0)

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

    # Drop the 'Adj Close' column from the dataframe
    df.drop('Adj Close', axis=1, inplace=True)

    # Save the updated dataframe back to the CSV file
    df.to_csv(f'new{file}', index=False)


def clean(size, file):
    stock_name = os.path.splitext(file)[0]
    
    # read the csv into a pandas DataFrame
    df = pd.read_csv(file, skiprows=range(1, size+1+19))
    
    # initialize balance and snapshot
    balance = 0
    snapshot = [balance]
    
    # initialize count of correct predictions and total predictions
    correct_predictions = 0
    total_predictions = 0
    
    # iterate over each row in the DataFrame
    try:
        for i in range(len(df)):
            # check if predicted buy is True
            if df.loc[i, 'Predicted buy'] == True:
                total_predictions += 1
                # calculate the new balance
                balance = balance + (1 * (df.loc[i, 'Close'] - df.loc[i+1, 'Close']))
                # increment the count of correct predictions if Buy is also True
                if df.loc[i, 'Buy'] == True:
                    correct_predictions += 1
            else:
                balance = balance + (1 * (- df.loc[i, 'Close'] + df.loc[i+1, 'Close']))
                total_predictions += 1
                if df.loc[i, 'Buy'] == False:
                    correct_predictions += 1
            
            # add the current balance to the snapshot
            snapshot.append(balance)
    except:
        print('something went wrong')
    
    # calculate the percentage of precision
    if total_predictions == 0:
        precision = 0
    else:
        precision = (correct_predictions / total_predictions) * 100
    
    # calculate the balance for each year
    n = len(snapshot)
    year1_bal = snapshot[n//3]
    year2_bal = snapshot[2*n//3]
    year3_bal = snapshot[-1]
    
    # plot the snapshot
    fig, ax = plt.subplots()
    ax.plot(snapshot)
    ax.axhline(y=0, color='black', linestyle='dotted')
    ax.text(n-1, year3_bal + 1, f"{year3_bal:.2f}", ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Balance', fontweight='bold')
    ax.set_title(f"Balance Over Time Size: {size}", fontweight='bold')
    # save the figure as an image
    plt.savefig(f"size{size}{stock_name}.png", dpi=300)
    
    # create a new DataFrame with the results
    results = pd.DataFrame({
        'Size': [size],
        'Precision': [precision],
        'Year 1 Balance': [year1_bal],
        'Year 2 Balance': [year2_bal],
        'Year 3 Balance': [year3_bal]
    })
    
    # return the results DataFrame
    return (results,snapshot)

import concurrent.futures

def process_data(size, file):
    stock_name = os.path.splitext(file)[0]

    fileName = f"size{size}{stock_name}.csv"

    # read the CSV file into a pandas DataFrame
    df = pd.read_csv(file, skiprows=range(1, 100-size+1+19))

    # remove the last row
    df = df.iloc[:-1]

    # loop over the range of indices with a step size of 100
    for i in range(0, len(df)-size, 1):
        try:
            # slice the DataFrame based on the current index and the next index
            X = df.iloc[i:i+size, 1:11].values
            y = df.iloc[i:i+size, 11].values

            # encode the target variable as integers
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

            # split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

            # standardize the data
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # perform LDA for dimensionality reduction
            lda = LinearDiscriminantAnalysis(n_components=1)
            X_train = lda.fit_transform(X_train, y_train)
            X_test = lda.transform(X_test)

            # define the parameter grid for grid search
            param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'poly', 'sigmoid']}
            

            # perform grid search to find the best hyperparameters
            grid_search = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0)
            grid_search.fit(X_train, y_train)
            
            # train the SVM classifier with the best hyperparameters
            classifier = grid_search.best_estimator_
            classifier.fit(X_train, y_train)

            # make predictions on the test data
            y_pred = classifier.predict(X_test)

            # predict the next value and save it to a CSV file
            next_X = df.iloc[i+size+1:i+size+2, 1:11].values
            next_X = sc.transform(next_X)
            next_X = lda.transform(next_X)
            next_y_pred = classifier.predict(next_X)
            next_y_pred = encoder.inverse_transform(next_y_pred)
            df.loc[i+size+1, 'Predicted buy'] = next_y_pred[0]
            df.to_csv(fileName, index=False)
        except Exception as e:
            print(e)
    
    time.sleep(2)
                    
    results = clean(size, fileName)[0]
        
    snapshots = clean(size, fileName)[1]
    
    return(results, snapshots)

def save_to_folder(folder_name):
    # create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # move all generated files to the folder
    for filename in os.listdir('.'):
        if filename.startswith(('size', 'result', 'snaps')):
            os.rename(filename, os.path.join(folder_name, filename))

def generate_results(file):
    sizes = list(range(10, 101, 1))

    all_results = []
    all_snapshots = []

    # set the number of processes to use
    NUM_PROCESSES = multiprocessing.cpu_count()
    
    # create the ProcessPoolExecutor with the specified number of processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = [executor.submit(process_data, size, file) for size in sizes]
    
        for future in concurrent.futures.as_completed(futures):
            results, snapshots = future.result()
            if results is not None:
                all_results.append(results)
            if snapshots is not None:
                all_snapshots.append(snapshots)

    # create a new DataFrame with all the results
    combined_results = pd.concat(all_results, ignore_index=True)
    snaps = pd.DataFrame(all_snapshots)

    # save the combined results to a CSV file
    stock_name = os.path.splitext(file)[0]
    combined_results.to_csv(f'{stock_name}_results.csv', index=False)
    snaps.to_csv(f'{stock_name}_snapshots.csv', index=False)

    # move all generated files to a folder
    save_to_folder(f'results_{stock_name}')

def run(file):
    add_technical_indicators(file)
    time.sleep(2)
    generate_results(f"new{file}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Works for any dataset between 2019-07-12 to 2023-01-03
    run("JNJ.csv")
    run("AAPL.csv")
    run("KO.csv")
    run("TSLA.csv")
    run("AMZN.csv")

