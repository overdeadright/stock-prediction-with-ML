import numpy as np
import csv
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import math
import pandas as pd
import sys
import os

def makeFile(file_output, cs, ds, stock):
    
    # Create a list to store the results
    results = ["stock"]
    
    # Loop through all the combinations of c and d
    for d in ds:
        for c in cs:
            # Add the results to the list
            results.append(f"d = {d}, c = {c}")

    if not os.path.exists(file_output):
        # Open a new CSV file in write mode
        with open(file_output, mode='w', newline='') as file:
        
            # Create a writer object
            writer = csv.writer(file)
        
            # Write the header row
            writer.writerow(results)

def ucb_with_daily_rewards_and_record(reward_file):
    
    print(reward_file)
    
    file_output = "results.csv"
    
    stock = reward_file.split(".")[0].split("_")[2]

    # Load the dataset from the specified CSV file
    dataset = pd.read_csv(reward_file)
    #dataset = dataset.iloc[:, 10:31]  # This line is commented out
    
    # Set some constants and initialize data storage variables
    N = len(dataset)  # Number of days in the dataset
    d = 9  # Number of slots
    
    D = [10]
    C = [1]
    outcome = [stock]
    
    makeFile(file_output, C, D, stock)
    
    for d in D:
        for c in C:
            #snapshot = []  # Stores the total reward achieved up to each day
            ads_selected = []  # Stores the ad selected on each day
            numbers_of_selections = [0] * d  # Tracks how many times each ad has been selected
            sums_of_rewards = [0] * d  # Tracks the sum of rewards for each ad
            total_reward = 0  # Total reward achieved over the entire dataset
            #arm = []  # Stores the ad that was chosen on each day
            
            # Loop over each day in the dataset
            for n in range(0, N):
                ad = 0  # Initialize the selected ad to 0
                max_upper_bound = 0  # Initialize the maximum UCB value to 0
                
                # Loop over each ad to calculate its UCB value
                for i in range(0, d):
                    if (numbers_of_selections[i] > 0):
                        average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                        delta_i = c * math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                        upper_bound = average_reward + delta_i
                    else:
                        upper_bound = 1e400  # Set a very large value for unselected ads
                        
                    # Update the selected ad if its UCB value is higher than the current maximum
                    if (upper_bound > max_upper_bound):
                        max_upper_bound = upper_bound
                        ad = i
                        
                # Record the selected ad and update data storage variables
                ads_selected.append(ad)
                numbers_of_selections[ad] = numbers_of_selections[ad] + 1
                reward = dataset.values[n, ad]
                sums_of_rewards[ad] = sums_of_rewards[ad] + reward
                total_reward = total_reward + reward
                
                
                # Record the current snapshot of total reward and the selected arm
                #snapshot.append(total_reward)
                #arm.append(ad)
            outcome.append(total_reward/d)
    # Read the contents of the CSV file and print them
    with open(file_output, mode='a', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)
    
        # Write the header row
        writer.writerow(outcome)

    
    # Print the snapshot and arm lists and return the total reward achieved over the entire dataset
    #print(snapshot)
    #print(arm)
    
    return(total_reward)

def run_algorithm(reward_file):
    # Run the Thompson Sampling algorithm
    arms = ucb_with_daily_rewards_and_record(reward_file)
    #print(arms[0])

    # Calculate the total sum and return it
    return arms

def formatter(file):
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
    
        # skip the header row
        next(csv_reader)
    
        # sort the data by the first column in descending order
        sorted_data = sorted(csv_reader, key=lambda row: int(row[0]), reverse=False)
    
        # remove the first column from the sorted data
        processed_data = [row[1:] for row in sorted_data]
    
        # transpose the data
        transposed_data = list(map(list, zip(*processed_data)))
    
        # add the last 1/3 of the rows to the transposed data list
        #num_rows = len(transposed_data)
        # SWAP IT BACK
        #last_third = transposed_data[:int(num_rows * 2 / 3)]
        #transposed_data = last_third
    
        # save the transposed data to a new CSV file
        with open(f'transposed_{file}', 'w', newline='') as new_csv_file:
            csv_writer = csv.writer(new_csv_file)
            csv_writer.writerows(transposed_data)  

def run(files):
    print(files)
    
    for file in files:
        formatter(file)

        # Define the file names for the reward and record files
        reward_file = f'transposed_{file}'
        record_file = f"record_{file}"
        stock = record_file.split(".")[0].split("_")[2]
        
        #print(stock)
        
        test = '''
        # Initialize an empty list to store the total sums        f
        or i in range(num_runs):
            output = nthompson_sampling(reward_file, record_file, slots, tokens)
            total_sums.append(output[1])
        resultSummary(total_sums)
        
        import sys
        sys.exit()
        '''
    
        run_algorithm(reward_file)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    files = ["difference_AAPL.csv", "difference_AMZN.csv", "difference_GOOGL.csv", 
             "difference_JNJ.csv", "difference_KO.csv"]
    total_sums = run(files)
    
    
    
    
    
    
    
    
    
    
