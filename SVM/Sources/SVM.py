import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from Clean import clean
import time


# setting
sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55,60,65,70,75, 80, 85,90, 95, 100]

all_results = []

all_snapshots = []

#file = "newJNJ.csv"
file = "newAAPL.csv"
#file = "newTSLA.csv"

for size in sizes:
    fileName = f"size{size}{file}"
    
    # read the CSV file into a pandas DataFrame
    df = pd.read_csv(file, skiprows=range(1, 100-size+1+19))
    
    # remove the last row
    df = df.iloc[:-1]
    
    # loop over the range of indices with a step size of 100
    for i in range(0, len(df)-size, 1):
        try:
            # slice the DataFrame based on the current index and the next index
            X = df.iloc[i:i+size, 1:12].values
            y = df.iloc[i:i+size, 12].values
            
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
            param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        
            # perform grid search to find the best hyperparameters
            grid_search = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
            grid_search.fit(X_train, y_train)
        
            # train the SVM classifier with the best hyperparameters
            classifier = grid_search.best_estimator_
            classifier.fit(X_train, y_train)
        
            # make predictions on the test data
            y_pred = classifier.predict(X_test)
        
            # predict the next value and save it to a CSV file
            next_X = df.iloc[i+size+1:i+size+2, 1:12].values
            next_X = sc.transform(next_X)
            next_X = lda.transform(next_X)
            next_y_pred = classifier.predict(next_X)
            next_y_pred = encoder.inverse_transform(next_y_pred)
            df.loc[i+size+1, 'Predicted buy'] = next_y_pred[0]
            df.to_csv(fileName, index=False)
        except Exception as e:
            print(e)
    
    time.sleep(2)
    
    output = clean(size, fileName)
    
    results = clean(size, fileName)[0]
    
    snapshots = clean(size, fileName)[1]
    
    # append the results to the list
    all_results.append(results)
    all_snapshots.append(snapshots)

# create a new DataFrame with all the results
combined_results = pd.concat(all_results, ignore_index=True)

snaps = pd.DataFrame(all_snapshots)

# save the combined results to a CSV file
combined_results.to_csv(f'result{file}', index=False)
snaps.to_csv('snaps.csv', index=False)


   
    