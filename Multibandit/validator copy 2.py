import csv

# initialize variables for storing the column sums and counts
first_half_sum = [0] * 3
first_half_count = 0
second_half_sum = [0] * 3
second_half_count = 0

files = ["difference_AAPL.csv", "difference_AMZN.csv", "difference_GOOGL.csv", 
         "difference_JNJ.csv", "difference_KO.csv", "difference_META.csv", 
         "difference_MSFT.csv", "difference_PG.csv", "difference_V.csv", "difference_WMT.csv"]

# loop through each file
for file in files:
    stock = file.split(".")[0].split("_")[1]
    filename = f"difference_{file}.csv"
    # open the CSV file
    with open(filename, 'r') as f:
        # create a CSV reader object
        reader = csv.reader(f)
        # initialize variables for storing column sums and counts for this file
        file_first_half_sum = [0] * 3
        file_first_half_count = 0
        file_second_half_sum = [0] * 3
        file_second_half_count = 0
        # iterate over the rows
        for row in reader:
            # convert the row elements to floats
            row = [float(x) for x in row[1:]]
            
            row = [float(x) for x in row]
            # sum up columns 1-3 for the first half and columns 4-6 for the second half
            for j, value in enumerate(row):
                if j < 3:
                    file_first_half_sum[j] += value
                    first_half_sum[j] += value
                    file_first_half_count += 1
                    first_half_count += 1
                else:
                    file_second_half_sum[j-3] += value
                    second_half_sum[j-3] += value
                    file_second_half_count += 1
                    second_half_count += 1
        # calculate the means for this file
        file_first_half_mean = [x / file_first_half_count for x in file_first_half_sum]
        file_second_half_mean = [x / file_second_half_count for x in file_second_half_sum]
        # write the means to the output file
        with open('output_both.csv', 'a') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([stock, file_first_half_mean, file_second_half_mean])
            
            
            
            
            
            
            
            
            
            