import cv2  # OpenCV
import requests
from pyzbar.pyzbar import decode

# --- Configuration ---
# Feature 1: API for Barcode Data (e.g., EAN, UPC, CODE128)
BARCODE_API_URL = "http://127.0.0.1:5000/barcode_data"
# Feature 2: API for QR Code Data
QRCODE_API_URL = "http://127.0.0.1:5000/qrcode_data"
# Feature 3: API for Image Upload
IMAGE_API_URL = "http://127.0.0.1:5000/upload_image"

# Filename to use when saving a captured image
IMAGE_FILE = "manual_capture.jpg"

# --- Helper Function: Features 1 & 2 ---
def send_data_to_api(url, code_data, code_type):
    """Sends text data (from a code) as JSON to the specified URL."""
    try:
        # Send the data in a JSON payload
        payload = {'data': code_data, 'type': code_type}
        response = requests.post(url, json=payload)
        print(f"Sent {code_type} data to {url}. Server response: {response.status_code}")
        # print(f"Response content: {response.text}") # Uncomment for debugging
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {url}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")

# --- Helper Function: Feature 3 ---
def send_image_to_api(url, frame):
    """Saves a frame as an image and uploads it to the specified URL."""
    # 1. Save the image to a file
    try:
        cv2.imwrite(IMAGE_FILE, frame)
        print(f"Image saved as {IMAGE_FILE}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return

    # 2. Send the file
    try:
        with open(IMAGE_FILE, 'rb') as f:
            files = {'file': (IMAGE_FILE, f, 'image/jpeg')}
            response = requests.post(url, files=files)
            print(f"Sent image to {url}. Server response: {response.status_code}")
            print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {url}")
    except Exception as e:
        print(f"An error occurred while sending image: {e}")

# --- Main Application Logic ---
def main():
    # 1. Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started.")
    print("Press 'c' to capture and send image.")
    print("Press 'q' to quit.")

    # This set stores the data of codes we've already sent
    # to prevent spamming the API every single frame.
    sent_codes = set()

    while True:
        # 2. Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # --- Handle Key Presses (Features 3 & Quit) ---
        # We check for keys every frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('c'):
            print("'c' pressed. Capturing and sending image...")
            # Feature 3: Send the current frame as an image
            send_image_to_api(IMAGE_API_URL, frame)

        # --- Handle Code Scanning (Features 1 & 2) ---
        barcodes = decode(frame)
        # Keep track of codes currently visible in the frame
        current_codes_in_frame = set()

        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            # Add data to our "currently visible" set
            current_codes_in_frame.add(barcode_data)
            # 4. A code was found! Check if it's new.
            if barcode_data not in sent_codes:
                print(f"Found new {barcode_type}! Data: {barcode_data}")
                if barcode_type == 'QRCODE':
                    # Feature 2: Send QR code data to its API
                    send_data_to_api(QRCODE_API_URL, barcode_data, barcode_type)
                else:
                    # Feature 1: Send Barcode data (EAN13, CODE128, etc.) to its API
                    send_data_to_api(BARCODE_API_URL, barcode_data, barcode_type)
                # Add this code to the "sent" list to avoid spam
                sent_codes.add(barcode_data)

            # Draw a box around the code regardless
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Clean up the sent_codes list
        # If a code was in the "sent" list but is no longer visible,
        # remove it, so it can be scanned again if it reappears.
        codes_to_remove = sent_codes - current_codes_in_frame
        for code in codes_to_remove:
            sent_codes.remove(code)

        # 8. Show the live camera feed
        cv2.imshow("Webcam Feed (Press 'c' to capture, 'q' to quit)", frame)

    # 10. Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()