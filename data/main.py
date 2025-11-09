
import csv
import random

# Define the filename
filename = 'barcodes.csv'
num_rows = 1000

# Generate the data and write to a CSV file
try:
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow(['barcode_id', 'type'])
        
        # Write the data rows
        for _ in range(num_rows):
            # Generate a 12-digit random barcode ID
            barcode_id = random.randint(100000000000, 999999999999)
            # Generate a random type (0, 1, or 2)
            barcode_type = random.randint(0, 2)
            
            # Write the row
            writer.writerow([barcode_id, barcode_type])
            
    print(f"Successfully generated {num_rows} barcode entries in '{filename}'.")

except Exception as e:
    print(f"An error occurred: {e}")