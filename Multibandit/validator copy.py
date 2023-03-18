import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import os


def nthompson_sampling(reward_file, record_file, slots = 1, tokens = 1, epsilon = 0.1):
    rewards = []

    # Load the rewards file into a list of lists
    with open(reward_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rewards.append([float(val) for val in row])
    
    # how many days we are running for
    num_trials = len(rewards)
    # n is total slots
    n = len(rewards[0])

    # Initialize the beta/normal distribution parameters with 1 success and 1 failure for each arm and each day
    num_successes = np.ones((n, num_trials))
    num_failures = np.ones((n, num_trials))
    
    track_balance = [0]

    # Run the n-Thompson Sampling algorithm for the specified number of trials
    for i in range(num_trials):
        samples = np.zeros(n)
        # For each arm, generate a random sample from its current beta/normal distribution and record it in the samples array
        for arm in range(n):
            # Choose one of 2 distribution
            samples[arm] = np.random.beta(num_successes[arm].sum(), num_failures[arm].sum())
            #samples[arm] = np.random.normal(num_successes[arm].sum() / (num_successes[arm].sum() + num_failures[arm].sum()), np.sqrt(num_successes[arm].sum() * num_failures[arm].sum() / (num_successes[arm].sum() + num_failures[arm].sum())**2))
        
        # Choose the top k arms with the highest samples as the selected arms for this trial
        if np.random.rand() < epsilon:
            # Choose a random arm
            chosen_arms = np.random.choice(n, slots, replace=False)
        else:
            # Choose the top slots arms with the highest samples
            chosen_arms = np.argpartition(samples, -slots)[-slots:]
            
        # How much total is split between each arms based on their distribution
        spending = split_tokens(tokens, samples[chosen_arms])
        spending = np.array(spending)
        
        # Determine the reward received for each chosen arm on this day
        rewards_this_day = rewards[i]
        rewards_received = np.array([rewards_this_day[chosen_arm] for chosen_arm in chosen_arms])
        
        rewards_received = np.multiply(rewards_received, spending)

        # Update the beta distribution parameters for each chosen arm based on the rewards received
        for j, chosen_arm in enumerate(chosen_arms):
            reward = rewards_received[j]
            if reward >= 0:
                num_successes[chosen_arm][i] += 1
            elif reward < 0:
                num_failures[chosen_arm][i] += 1
        
        #Write to record
        #writeRecord(record_file, rewards_received, n, chosen_arms, spending, slots)
        # Track balance overtime
        track_balance.append(track_balance[-1] + sum(rewards_received))
    
    # Return the indices of the k arms with the highest total number of successes plus failures across all days
    indices = np.argpartition(num_successes.sum(axis=1) + num_failures.sum(axis=1), -slots)[-slots:]
    return indices, track_balance[-1]


def writeRecord(record_file, rewards_received, n, chosen_arms, spending, slots):
    # Initialize the record file
    if not os.path.isfile(record_file):
        with open(record_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["trial"] + [f"arm {i}" for i in range(1, n+1)] + [f"reward {i}" for i in range(1, slots+1)] + ["total"])
        
    lst = [int(arm in chosen_arms) for arm in range(n)]
    num_ones = 0
    for i in range(len(lst)):
        if lst[i] == 1:
            lst[i] = spending[num_ones]
            num_ones += 1
            if num_ones == len(spending):
                break
            
    with open(record_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        row = [i+1] + lst + list(rewards_received) + [sum(rewards_received)]
        writer.writerow(row)
        
def stats_zero(numbers):    
    # Count the number of elements greater than 0
    num_greater_than_zero = len(np.where(np.array(numbers) > 0)[0])

    # Calculate the probability
    prob_greater_than_zero = num_greater_than_zero / len(numbers)
    
    print("Probability of value being greater than 0: {:.2f}".format(prob_greater_than_zero))
    
    return(prob_greater_than_zero)

hello = '''
def formatter(file):
    # open the original CSV file
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
    
        # save the transposed data to a new CSV file
        with open(f'transposed_{file}', 'w', newline='') as new_csv_file:
            csv_writer = csv.writer(new_csv_file)
            csv_writer.writerows(transposed_data)
            
    time.sleep(1)'''
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

    
# Split total amount of tokens amoungs the slots
def split_tokens(tokens, proportions):
    # Calculate the multiplier to scale the proportions to the total
    total_proportions = sum(proportions)
    multiplier = tokens / total_proportions

    # Multiply each proportion by the multiplier and round to the nearest integer
    values = [round(proportion * multiplier) for proportion in proportions]

    # Distribute any remaining amount evenly among the values
    remaining = tokens - sum(values)
    highest_to_lowest = sorted(range(len(proportions)), key=lambda i: -proportions[i])
    lowest_to_highest = sorted(range(len(proportions)), key=lambda i: proportions[i])
    while True:
        for i, j in zip(highest_to_lowest, lowest_to_highest):
            if remaining > 0:
                values[i] += 1
                remaining -= 1
            elif remaining < 0:
                values[j] += 1
                remaining += 1
            else:
                break
        if remaining == 0:
            break

    return values

def run(files):
    # Define the number of trials, arms, and runs
    slots = 10
    num_runs = 1000
    
    #token > slots
    tokens = 10
    
    for file in files:
        formatter(file)

        # Define the file names for the reward and record files
        reward_file = f'transposed_{file}'
        record_file = f"record_{file}"
        
        # Initialize an empty list to store the total sums
        total_sums = []
        for i in range(num_runs):
            output = nthompson_sampling(reward_file, record_file, slots, tokens)
            total_sums.append(output[1])
        resultSummary(total_sums)
        
        import sys
        sys.exit()
    
        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit num_runs tasks to the executor and store the future objects in a list
            futures = [executor.submit(nthompson_sampling, reward_file, record_file, slots, tokens) for i in range(num_runs)]
    
            # Initialize a counter variable to keep track of the current iteration number
            current_iteration = 1
            
            # Iterate over the futures as they complete and append the total sum to the list
            for future in concurrent.futures.as_completed(futures):
                total_sums.append(future.result())
                
                # Print the current iteration number and increment the counter
                print("Iteration {}/{}".format(current_iteration, num_runs))
                current_iteration += 1
        
        resultSummary(total_sums)

def resultSummary(total_sums):
    mean = np.mean(total_sums)
    median = np.median(total_sums)
    stddev = np.std(total_sums)
    minval = np.min(total_sums)
    maxval = np.max(total_sums)
    q1 = np.percentile(total_sums, 25)
    q3 = np.percentile(total_sums, 75)
    
    stats_zero(total_sums)
    
    # Create a histogram of the total sums
    plt.hist(total_sums, bins=30)
    plt.title("Distribution of Total Sums")
    plt.xlabel("Total Sum")
    plt.ylabel("Frequency")
    plt.show()
        
    print("Mean: {:.2f}".format(mean))
    print("Median: {:.2f}".format(median))
    print("Standard deviation: {:.2f}".format(stddev))
    print("Minimum value: {:.2f}".format(minval))
    print("Maximum value: {:.2f}".format(maxval))
    print("25th percentile: {:.2f}".format(q1))
    print("75th percentile: {:.2f}".format(q3))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    total_sums = run(["difference_WMT.csv"])
    
