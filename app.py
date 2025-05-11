import base64
import cv2
import numpy as np
import logging
from flask import Flask, jsonify, redirect, render_template, Response, request, url_for
import os
import threading

# Initialize Flask app
app = Flask(__name__)

# Directory to save processed images
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

# Load the pre-trained Haar Cascade for eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def detect_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Crop face region
        face_region = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_region)
        
        # If at least one eye is detected, return True
        if len(eyes) > 0:
            return True
    
    # If no eyes detected
    return False

# Function to process the image
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_img = cv2.equalizeHist(gray)
    edges = cv2.Canny(enhanced_img, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.dilate(edges, kernel, iterations=1)
    return processed_image

# Function to calculate pressure
def calculate_pressure(length, density):
    # Define the original pressure range and the desired range
    original_min = 70
    original_max = 120
    desired_min = 10
    desired_max = 21

    # Calculate the original pressure using the previous formula
    original_pressure = (0.5 * length * density) / 10
    logging.debug(f"Original Pressure: {original_pressure}")

    # Scale the original pressure to the desired range using linear transformation
    pressure = desired_min + (original_pressure - original_min) * (desired_max - desired_min) / (original_max - original_min)
    logging.debug(f"Scaled Pressure: {pressure}")

    # Ensure the pressure stays within the desired range (clipping)
    pressure = max(min(pressure, desired_max), desired_min)

    return round(pressure, 2)

# Function to provide warnings based on pressure
def get_pressure_warning(pressure):
    if pressure > 21:
        return "High Pressure", "danger"
    elif pressure < 12:
        return "Low Pressure", "warning"
    else:
        return "Normal Pressure", "success"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ready', methods=['POST'])
def ready_for_test():
    return redirect(url_for('index'))

@app.route('/Eye')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "Error: Camera could not be opened"
    
    def capture():
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    # Start capturing in a separate thread to avoid blocking
    thread = threading.Thread(target=capture)
    thread.start()

@app.route('/capture', methods=['POST'])
def capture_and_process():
    try:
        data = request.get_json()
        img_data = data['image']

        # Check if the image data starts with the correct prefix
        if not img_data.startswith('data:image/jpeg;base64,'):
            return jsonify({
                'pressure': None,
                'message': "Invalid image format. Please upload a valid JPEG image.",
                'alert_type': "warning",
                'image_path': None
            })

        # Extract the base64 part of the image data (after the prefix)
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Ensure eye detection is successful (no full face accepted)
        if not detect_eye(image):
            return jsonify({
                'pressure': None,
                'message': "No eye detected or image contains full face. Please upload an eye-focused image.",
                'alert_type': "warning",
                'image_path': None
            })

        processed_image = process_image(image)
        processed_image_path = os.path.join('static', 'images', 'processed_image.jpg')
        cv2.imwrite(processed_image_path, processed_image)

        vessel_length = np.sum(processed_image > 0)
        vessel_density = vessel_length / processed_image.size
        logging.debug(f"Vessel Length: {vessel_length}")
        logging.debug(f"Vessel Density: {vessel_density}")
        
        if vessel_length == 0 or vessel_density == 0:
            return jsonify({
                'pressure': None,
                'message': "Error: Vessel length or density calculation is zero. Image may not contain sufficient data.",
                'alert_type': "danger",
                'image_path': None
            })

        estimated_pressure = calculate_pressure(vessel_length, vessel_density)
        message, alert_type = get_pressure_warning(estimated_pressure)

        return jsonify({
            'pressure': estimated_pressure,
            'message': message,
            'alert_type': alert_type,
            'image_path': 'images/processed_image.jpg'
        })

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({
            'pressure': None,
            'message': f"An unexpected error occurred: {str(e)}",
            'alert_type': "danger",
            'image_path': None
        })

@app.route('/stop')
def stop_camera():
    cv2.destroyAllWindows()
    return "Camera stopped"

if __name__ == '__main__':
    app.run(debug=True)
