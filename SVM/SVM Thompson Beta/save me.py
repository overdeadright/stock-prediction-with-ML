import csv
import sys

files = ["difference_AAPL.csv", "difference_AMZN.csv", "difference_GOOGL.csv", 
         "difference_JNJ.csv", "difference_KO.csv", "difference_META.csv", 
         "difference_MSFT.csv", "difference_PG.csv", "difference_V.csv", "difference_WMT.csv"]

for file in files:
    stock = file.split(".")[0].split("_")[1]
    filename = file
    
    # open the CSV file
    with open(filename, 'r') as f:
        # create a CSV reader object
        reader = csv.reader(f)
        # initialize variables for storing column sums and counts for this file
        year1 = []
        year2 = []
        year3 = []
        # iterate over the rows
        for row in reader:
            # convert the row elements to floats
            row = [float(x) for x in row[1:]]

            n = len(row)
            third = n // 3

            sum_first_third = sum(row[:third])
            sum_second_third = sum(row[third:2*third])
            sum_last_third = sum(row[2*third:])

            year1.append(sum_first_third)
            year2.append(sum_second_third)
            year3.append(sum_last_third)
                    
        # calculate the means for this file
        import statistics

        year1 = statistics.mean(year1)
        year2 = statistics.mean(year2)
        year3 = statistics.mean(year3)
        # write the means to the output file
        with open('output_both2.csv', 'a') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([stock, year1, year2, year3])