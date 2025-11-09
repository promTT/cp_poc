import csv
import random  # <-- We need this module for random sampling
from barcode import get_barcode_class
from barcode.writer import ImageWriter
import os

# --- Configuration ---
CSV_FILENAME = 'data/mockData.csv'
OUTPUT_FOLDER = 'barcodes'
NUMBER_TO_PICK = 10
# ---------------------

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Get the barcode type
Code128 = get_barcode_class('code128')

print(f"Reading from {CSV_FILENAME} to find all IDs...")

# --- Step 1: Read ALL IDs from the CSV into a list ---
all_ids = []
try:
    with open(CSV_FILENAME, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if 'barcode_id' in row and row['barcode_id']:  # Check if 'id' column exists and is not empty
                all_ids.append(row['barcode_id'])
            else:
                print("Skipping a row with missing or empty ID.")
                
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILENAME}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred reading the file: {e}")
    exit()

if not all_ids:
    print("No valid IDs were found in the file. Exiting.")
    exit()

print(f"Found {len(all_ids)} total IDs.")

# --- Step 2: Randomly select 10 IDs ---

# Handle case where there are fewer than 10 IDs in the file
if len(all_ids) <= NUMBER_TO_PICK:
    print(f"Warning: Only found {len(all_ids)} IDs. Will use all of them.")
    selected_ids = all_ids
else:
    # This is the main random selection
    selected_ids = random.sample(all_ids, NUMBER_TO_PICK)
    print(f"Randomly selected {len(selected_ids)} IDs to process.")


# --- Step 3: Loop through ONLY the 10 selected IDs and create barcodes ---
count = 0
for user_id in selected_ids:
    try:
        # Create the barcode object
        my_barcode = Code128(user_id, writer=ImageWriter())
        
        # Save the barcode image
        filename = os.path.join(OUTPUT_FOLDER, f"{user_id}")
        my_barcode.save(filename)
        
        print(f"  -> Generated barcode for: {user_id} (saved as {filename}.png)")
        count += 1
        
    except Exception as e:
        print(f"Could not create barcode for ID '{user_id}'. Error: {e}")

print(f"\nDone. Generated {count} random barcodes in the '{OUTPUT_FOLDER}' folder.")