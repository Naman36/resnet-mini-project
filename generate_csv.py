import csv
# Function for generating the csv file for submission


# Output File
output_csv_file = './predictions/predictions.csv'


def generate_csv(ids, predictions):
# Writing to the CSV file
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'Label'])
        # Write the ID and Label pairs
        for id, label in zip(ids, predictions):
            writer.writerow([id, label])

    print(f"CSV file '{output_csv_file}' has been created with ID and Label columns.")
