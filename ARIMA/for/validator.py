import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import os

def nthompson_sampling(reward_file, record_file, slots = 1, tokens = 1):
    """
    n: number of arms
    reward_file: filename for the rewards file, where each row contains the expected rewards for a particular day
    num_trials: number of trials to run the algorithm for
    k: number of arms to select
    record_file: filename to store the record of chosen arms and rewards
    """
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
            #samples[arm] = np.random.beta(num_successes[arm].sum(), num_failures[arm].sum())
            samples[arm] = np.random.normal(num_successes[arm].sum() / (num_successes[arm].sum() + num_failures[arm].sum()), np.sqrt(num_successes[arm].sum() * num_failures[arm].sum() / (num_successes[arm].sum() + num_failures[arm].sum())**2))
        
        # Choose the top k arms with the highest samples as the selected arms for this trial
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

def stats_zero(numbers):    
    # Count the number of elements greater than 0
    num_greater_than_zero = len(np.where(np.array(numbers) > 0)[0])

    # Calculate the probability
    prob_greater_than_zero = num_greater_than_zero / len(numbers)
    
    #print("Probability of value being greater than 0: {:.2f}".format(prob_greater_than_zero))
    
    return(prob_greater_than_zero)

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

def resultSummary(total_sums, stock):
    mean = np.mean(total_sums)
    median = np.median(total_sums)
    stddev = np.std(total_sums)
    minval = np.min(total_sums)
    maxval = np.max(total_sums)
    q1 = np.percentile(total_sums, 25)
    q3 = np.percentile(total_sums, 75)
    
    stats = stats_zero(total_sums)
    
    # Create a histogram of the total sums
    plt.hist(total_sums, bins=30)
    plt.title(f"Distribution of Total Sums for {stock}")
    plt.xlabel("Total Sum")
    plt.ylabel("Frequency")
    plt.axvline(x=0, linestyle='dotted', color='grey')
    plt.savefig(f'total_sums_distribution_{stock}.png', dpi=300)
    plt.show()
    plt.clf()
        
    # Check if file exists
    filename = 'output.csv'

    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        # Write header row if file doesn't exist
        if not file_exists:
            writer.writerow(['Stock', 'Mean', 'Median', 'Standard deviation', 'Minimum value', 'Maximum value', '25th percentile', '75th percentile', 'Odds'])
            
        # Write data row
        writer.writerow([stock, "{:.2f}".format(mean), "{:.2f}".format(median), "{:.2f}".format(stddev), "{:.2f}".format(minval), "{:.2f}".format(maxval), "{:.2f}".format(q1), "{:.2f}".format(q3), stats])
    

def run(files):
    # Define the number of trials, arms, and runs
    slots = 10
    num_runs = 10000
    
    #token > slots
    tokens = 10
    
    for file in files:
        formatter(file)

        # Define the file names for the reward and record files
        reward_file = f'transposed_{file}'
        record_file = f"record_{file}"
        stock = record_file.split(".")[0].split("_")[2]
        
        print(stock)

    
        total_sums = []
        
        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit num_runs tasks to the executor and store the future objects in a list
            futures = [executor.submit(nthompson_sampling, reward_file, record_file, slots, tokens) for i in range(num_runs)]
    
            # Initialize a counter variable to keep track of the current iteration number
            current_iteration = 1
            
            # Iterate over the futures as they complete and append the total sum to the list
            for future in concurrent.futures.as_completed(futures):
                total_sums.append(future.result()[1])
                
                # Print the current iteration number and increment the counter
                print("Iteration {}/{}".format(current_iteration, num_runs))
                #print(total_sums[-1])
                current_iteration += 1
        
        total_sums_scaled = [x / tokens for x in total_sums]
        resultSummary(total_sums_scaled, stock)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import glob

    pattern = "/Users/kayn/Desktop/ARIMA/for/difference_*.csv"  # pattern to match files with a certain prefix and extension
    files = [os.path.basename(file) for file in glob.glob(pattern)]  # list of file names (without directory) that match the pattern
    print(files)
    total_sums = run(files)
    
