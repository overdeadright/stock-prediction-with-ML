import pandas as pd
import time
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

def process_data(size, file):
    # Where files will be stored
    stock_name = getStockName(f"preprocessed_{file}")
    
    # In the folder
    file_name = f"{stock_name}_Folder/{size}_{stock_name}.csv"
    
    # Read the CSV file into a DataFrame, including the header row
    df = pd.read_csv("preprocessed_AAPL.csv", header=0)

    # Remove the first 5 rows from the DataFrame
    df = df.iloc[100-size:]

    # remove the last row
    df = df.iloc[:-1]

    # loop over the range of indices with a step size of 100
    for i in range(0, len(df)-size, 1):
        try:
            X = df.iloc[:, 1:-1].values  # select all rows and all columns except the first and last columns
            y = df.iloc[:, -1].values   # select all rows and the last column
           
            # encode the target variable as integers
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

            # split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2023)

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

            # predict the next value as True or False
            next_X = df.iloc[i+size:i+size+1, 1:-1].values
            next_X = sc.transform(next_X)
            next_X = lda.transform(next_X)
            next_y_pred = classifier.predict(next_X)
            next_y_pred = encoder.inverse_transform(next_y_pred)
            
            df.loc[i+size, 'Predicted buy'] = next_y_pred[0]
            df.to_csv(file_name, index=False)
        except Exception as e:
            print(e)
    
    time.sleep(2)
                    
    results = clean(size, fileName)[0]
        
    snapshots = clean(size, fileName)[1]
    
    return(results, snapshots)


def getStockName(file):
    # In format "preprocessed_{StockName}.csv"
    parts = file.split("_")   # split the string by "_" characters
    ticker = parts[1].split(".")[0]   # extract the second part and remove the ".csv" extension
    return ticker

def run(file):
    Preprocessor(file)
    # Create a file
    # create the new directory if it doesn't exist
    folder = getStockName(f"preprocessed_{file}") + "_Folder"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    generate_results(f"preprocessed_{file}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    #run("JNJ.csv")
    run("AAPL.csv")
    #run("KO.csv")
    #run("TSLA.csv")
    #run("AMZN.csv")
    
    