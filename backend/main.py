import os
import csv
import random
from flask import Flask, request, jsonify

# --- NEW AI MODEL IMPORTS ---
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- Configuration ---
app = Flask(__name__)
CSV_FILE = '../data/mockData.csv'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- NEW AI MODEL CONFIGURATION ---
MODEL_PATH = "model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224, 224) # 224x224 pixels

# 
# [!!] CRITICAL: YOU MUST UPDATE THIS LIST [!!]
#
# This list MUST match the exact order of classes your model was trained on.
# The model outputs a number (like '0'), which we map to the 0th item 
# in this list ('battery').
#
TARGET_CLASSES = [
    'battery', 
    'cardboard', 
    'glass', 
    'metal', 
    'paper', 
    'plastic',
    'trash' # Example: Add all your class names here
]
# --- END AI CONFIGURATION ---


# --- NEW AI HELPER FUNCTIONS ---

def get_transforms(img_size):
    """
    Returns the image transformations.
    NOTE: These MUST match the transformations used during training.
    The normalization values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]
    are standard for models pre-trained on ImageNet.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_trained_model(model_path, num_classes, device):
    """
    Loads a pre-trained ResNet-18 model and replaces its final
    layer, then loads your saved weights (`model.pth`).
    """
    try:
        # Load a pre-trained ResNet-18 model
        model = models.resnet18(pretrained=True)
        
        # Get the number of input features for the final layer
        num_ftrs = model.fc.in_features
        
        # Replace the final layer to match your number of classes
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load your saved weights
        # map_location=device ensures it works even if you trained on a GPU
        # and are now running on a CPU.
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        print(f"[AI Model] Model '{model_path}' loaded successfully on {device}.")
        return model.to(device)
    except FileNotFoundError:
        print(f"[AI Model] Error: Model file not found at '{model_path}'.")
        print("Please make sure 'model.pth' is in the same directory as 'app.py'.")
        return None
    except Exception as e:
        print(f"[AI Model] Error loading model: {e}")
        return None

def predict_image_from_path(model, image_path, device, target_classes, img_size):
    """
    This is your prediction function, adapted for the server.
    It takes the path to an image, processes it, and returns the
    predicted class name as a string.
    """
    try:
        # 1. Open the image file
        img = Image.open(image_path).convert('RGB')
        
        # 2. Apply the transformations
        transform = get_transforms(img_size)
        # Add a batch dimension (models expect a batch of images)
        img_tensor = transform(img).unsqueeze(0)
        
        # 3. Move tensor to the correct device
        img_tensor = img_tensor.to(device)
        
        # 4. Get prediction
        with torch.no_grad(): # Disable gradient calculation
            outputs = model(img_tensor)
            
            # 5. Get the class index with the highest score
            _, preds = torch.max(outputs, 1)
            predicted_index = preds.item()
        
        # 6. Map the index to the class name
        predicted_class = target_classes[predicted_index]
        return predicted_class
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- END AI HELPER FUNCTIONS ---


# --- LOAD THE MODEL ON STARTUP ---
# This code runs ONCE when you start `python app.py`
model = load_trained_model(MODEL_PATH, len(TARGET_CLASSES), DEVICE)
# ---


# --- API Endpoint 1: Find Type by Barcode/QR Code Data ---
# (This section is unchanged from the previous version)
@app.route('/barcode_data', methods=['POST'])
@app.route('/qrcode_data', methods=['POST'])
def find_type_by_data():
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            return jsonify({'status': 'error', 'message': 'Missing data field'}), 400

        code_id = request_data['data']
        print(f"[Data API] Received request for ID: {code_id}")

        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            
            for row in reader:
                if row[0] == code_id:
                    trash_type = row[1]
                    print(f"[Data API] ID found. Type: {trash_type}")
                    return jsonify({
                        'status': 'success', 
                        'found': True,
                        'id': code_id,
                        'trash_type': trash_type
                    })

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


# --- API Endpoint 2: Find Type by Image (NOW USES REAL AI) ---
@app.route('/upload_image', methods=['POST'])
def find_type_by_image():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        # 1. Save the uploaded file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(f"[Image API] Image saved to {file_path}")

        # 2. Call the REAL AI model for prediction
        print(f"[AI Model] Running prediction on {file_path}...")
        
        # Check if model loaded successfully
        if model is not None:
            # This is the code block you provided!
            predicted_class = predict_image_from_path(
                model, 
                file_path,  # Use the path of the file we just saved
                DEVICE, 
                TARGET_CLASSES, 
                IMG_SIZE
            )
            
            if predicted_class:
                print(f"[AI Model] Result = {predicted_class}")
                trash_type = predicted_class
            else:
                trash_type = "Unknown (Prediction failed)"
        else:
            print("[AI Model] Prediction skipped because model is not loaded.")
            trash_type = "Unknown (Model not loaded)"

        # 3. Return the AI's prediction
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
    print("Starting Flask server on http://124.0.0.1:5000")
    # We set debug=False because it can cause issues with loading
    # the model twice. If you are only changing API logic,
    # you can set debug=True, but 'False' is safer for production.
    app.run(debug=False, port=5000, host='127.0.0.1')