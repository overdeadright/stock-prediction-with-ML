import pandas as pd
import matplotlib.pyplot as plt

def clean(size, file):
    
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
    plt.savefig(f"AAPL{size}.png")
    plt.show()
    
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

