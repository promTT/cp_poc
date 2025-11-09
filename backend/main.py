import os
import csv
import time
import random
from flask import Flask, request, jsonify

# --- Configuration ---
app = Flask(__name__)
CSV_FILE = 'data/mockData.csv'
UPLOAD_FOLDER = 'uploads'
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- (Mock) AI Model ---
# This function SIMULATES your AI model.
# In a real app, you would replace this with code that
# loads your model (TensorFlow, PyTorch) and runs a prediction.
def get_ai_prediction(image_path):
    """
    Simulates a heavy AI model processing an image.
    """
    print(f"[AI Model] Processing image: {image_path}...")
    # Simulate AI processing time (e.g., 0.5 to 2 seconds)
    time.sleep(random.uniform(0.5, 2.0))
    
    # Simulate a random prediction
    possible_types = [
        "Plastic Bottle", 
        "Aluminum Can", 
        "Paper", 
        "General Waste",
        "Glass"
    ]
    prediction = random.choice(possible_types)
    print(f"[AI Model] Prediction: {prediction}")
    return prediction

# --- API Endpoint 1: Find Type by Barcode/QR Code Data ---
# This single function handles both /barcode_data and /qrcode_data
# from your client script.
@app.route('/barcode_data', methods=['POST'])
def find_type_by_data():
    try:
        # 1. Get the JSON data sent from your client
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            return jsonify({'status': 'error', 'message': 'Missing data field'}), 400

        code_id = request_data['data']
        print(f"[Data API] Received request for ID: {code_id}")

        # 2. Read the CSV file to find the matching ID
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader, None) 
            
            for row in reader:
                # Check if the ID in the first column matches
                if row[0] == code_id:
                    trash_type = row[1]
                    print(f"[Data API] ID found. Type: {trash_type}")
                    # 3. Return the type
                    return jsonify({
                        'status': 'success', 
                        'found': True,
                        'id': code_id,
                        'trash_type': trash_type
                    })

        # 3. If no match was found after checking the whole file
        print("[Data API] ID not found in CSV.")
        return jsonify({
            'status': 'success', 
            'found': False,
            'id': code_id,
            'message': 'ID not found in database'
        }), 404

    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found!")
        return jsonify({'status': 'error', 'message': 'Server database not found'}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- API Endpoint 2: Find Type by Image ---
# This handles the image upload from pressing 'c' in your client.
@app.route('/upload_image', methods=['POST'])
def find_type_by_image():
    try:
        # 1. Check if a file was sent
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in request'}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        # 2. Save the file to our 'uploads' folder
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(f"[Image API] Image saved to {file_path}")

            # 3. Call the "AI model" with the path to the saved image
            trash_type = get_ai_prediction(file_path)

            # 4. Return the AI's prediction
            return jsonify({
                'status': 'success',
                'filename': filename,
                'trash_type': trash_type
            })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Run the Server ---
if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    # debug=True allows the server to auto-reload when you save changes
    app.run(debug=True, port=5000)