import pandas as pd
import time
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
import numpy as np
from preprocessor import Preprocessor
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import warnings
import preprocessor
from tqdm import tqdm

warnings.filterwarnings("ignore")

def getStockName(file):
    # In format "preprocessed_{StockName}.csv"
    parts = file.split("_")   # split the string by "_" characters
    ticker = parts[1].split(".")[0]   # extract the second part and remove the ".csv" extension
    return ticker

def SVM(file, size):
    # Where files will be stored
    stock_name = getStockName(f"{file}")
    folder_name = f"{stock_name}_Folder"
    file_name = f"{folder_name}/{size}_{stock_name}.csv"   
    
    # Read the CSV file into a DataFrame, including the header row
    df = pd.read_csv(file, header=0)
    
    preprocessor.max_size

    # Remove the first 100-size rows from the DataFrame
    df = df.drop(index=range(preprocessor.max_size-size))
    
    # To save a file later on
    df_copy = df.copy().iloc[size:]
    predicted_buy = []
    
    # Balance is how much money you have at the end
    # The other two keep track of balance and difference is per day
    balance = 0
    balance_history = []
    difference_history = []

    # loop over the range of indices with a step size of 100
    for i in range(0, len(df)-size, 1):
        try:
            X_df = df.iloc[i:size+i, 1:-1]
            X = X_df.values  # select all rows and all columns except the first and last columns
            y = df.iloc[i:size+i:, -1].values   # select all rows and the last column
            
            # encode the target variable as integers
            encoder = OrdinalEncoder()
            y = encoder.fit_transform(y.reshape(-1, 1)).ravel()
    
            # split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1052001)
    
            # standardize the data
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            
            # perform LDA for dimensionality reduction
            lda = LinearDiscriminantAnalysis(n_components=1)  
            X_train = lda.fit_transform(X_train, y_train)
    
            # define the parameter grid for grid search
            param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'poly', 'sigmoid']}
            
            # perform grid search to find the best hyperparameters
            grid_search = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
            grid_search.fit(X_train, y_train)
            
            # train the SVM classifier with the best hyperparameters
            classifier = grid_search.best_estimator_
            classifier.fit(X_train, y_train)
    
            # predict the next value as True or False
            next_X_row = df.iloc[i+size:i+size+1, 1:-1]
            next_X = next_X_row.values
            next_X = sc.transform(next_X)
            next_X = lda.transform(next_X)
            next_y_pred = classifier.predict(next_X)
            next_y_pred = encoder.inverse_transform([next_y_pred])[0][0]
            
            # Calculate money earn and lost
            next_y_real = df.iloc[i+size:i+size+1, -1].values[0]
            next_close = next_X_row["Close"].values[0]
            predicted_buy.append(next_y_pred)
            
            next_next_close = 0

            try: 
                next_next_close = df.iloc[i+size+1:i+size+2, 1:-1]["Close"].values[0]
            except:
                break
                #nothing
    
            if next_y_real == next_y_pred:
                # Made money
                differece = abs(next_next_close - next_close)
                balance += differece
                balance_history.append(balance)
                difference_history.append(differece)
            else:
                # Lost money
                differece = -abs(next_next_close - next_close)
                balance += differece
                balance_history.append(balance)
                difference_history.append(differece)
                
            
        except Exception as e:
            # Shit happens
            predicted_buy.append("NONE")
            differece = 0
            balance += differece
            balance_history.append(balance)
            difference_history.append(differece)
            print(e)
    
    # Save values
    saveDifferences(folder_name, size, difference_history, stock_name)
    saveBalances(folder_name, size, balance_history, stock_name)
    plot_balance_history(folder_name, size, balance_history, stock_name)

    df_copy['Predicted buy'] = predicted_buy
    df_copy.to_csv(file_name, index=False)
    
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
    
    for size in sizes:
        SVM(f"preprocessed_{file}", size)
    
    sys.exit()


    # create a list to store the futures of the concurrent tasks
    futures = []
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for size in sizes:
            # submit the task to the executor and store the future object
            future = executor.submit(SVM, f"preprocessed_{file}", size)
            futures.append(future)
        
        # use tqdm to create a progress bar for the futures list
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            # check if the future returned an exception and re-raise it if it did
            if future.exception() is not None:
                raise future.exception()

if __name__ == '__main__':    
    multiprocessing.freeze_support()
    files = ["GME.csv"]

    # Run the code
    run_concurrent(files)


            
            
            
            
            
            
            