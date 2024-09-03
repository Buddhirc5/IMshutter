from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import cv2
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)

# Global variables to store the processed image and history for undo/redo
processed_img = None
undo_stack = []
redo_stack = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global processed_img, undo_stack, redo_stack
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if file:
        # Read the image using OpenCV
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        processed_img = img.copy()
        undo_stack = [processed_img.copy()]  # Initialize undo stack
        redo_stack = []  # Clear redo stack
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Invalid file.'}), 400

@app.route('/rotate', methods=['POST'])
def rotate_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        angle = 90  # Fixed angle of 90 degrees
        (h, w) = processed_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(processed_img, M, (w, h))
        processed_img = rotated
        undo_stack.append(processed_img.copy())  # Save state for undo
        redo_stack.clear()  # Clear redo stack
    return jsonify({'success': True}), 200

@app.route('/invert', methods=['POST'])
def invert_colors():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        processed_img = cv2.bitwise_not(processed_img)
        undo_stack.append(processed_img.copy())  # Save state for undo
        redo_stack.clear()  # Clear redo stack
    return jsonify({'success': True}), 200

@app.route('/crop', methods=['POST'])
def crop_image():
    global processed_img, undo_stack, redo_stack
    width = int(request.form.get('width', 100))
    height = int(request.form.get('height', 100))
    if processed_img is not None:
        h, w = processed_img.shape[:2]
        start_x = max(w // 2 - width // 2, 0)
        start_y = max(h // 2 - height // 2, 0)
        cropped_img = processed_img[start_y:start_y + height, start_x:start_x + width]
        processed_img = cropped_img
        undo_stack.append(processed_img.copy())  # Save state for undo
        redo_stack.clear()  # Clear redo stack
    return jsonify({'success': True}), 200

@app.route('/undo', methods=['POST'])
def undo():
    global processed_img, undo_stack, redo_stack
    if len(undo_stack) > 1:
        redo_stack.append(undo_stack.pop())  # Move current state to redo stack
        processed_img = undo_stack[-1].copy()  # Revert to the last state
    return jsonify({'success': True}), 200

@app.route('/redo', methods=['POST'])
def redo():
    global processed_img, undo_stack, redo_stack
    if redo_stack:
        undo_stack.append(redo_stack.pop())  # Move last redo state to undo stack
        processed_img = undo_stack[-1].copy()  # Set to the redo state
    return jsonify({'success': True}), 200

@app.route('/save', methods=['POST'])
def save_image():
    global processed_img
    if processed_img is not None:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg', as_attachment=True, download_name='edited_image.jpg')
    return jsonify({'error': 'No image to save.'}), 400

@app.route('/get_image')
def get_image():
    global processed_img
    if processed_img is not None:
        _, buffer = cv2.imencode('.jpg', processed_img)
        encoded_image = buffer.tobytes()
        return send_file(BytesIO(encoded_image), mimetype='image/jpeg')
    return jsonify({'error': 'No image to display.'}), 400


if __name__ == '__main__':
    app.run(debug=True)
