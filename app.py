from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global variables to store the processed image and history for undo/redo
processed_img = None
undo_stack = []
redo_stack = []

# Load the pre-trained DeepLabV3 model for segmentation
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set model to evaluation mode

# Check if CUDA is available and use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the image transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization parameters for ImageNet
                         std=[0.229, 0.224, 0.225]),
])

@app.route('/color', methods=['POST'])
def change_to_color():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image converted to color'}), 200

@app.route('/black_white', methods=['POST'])
def change_to_black_white():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        _, bw_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image converted to black and white'}), 200

@app.route('/grayscale', methods=['POST'])
def change_to_grayscale():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image converted to grayscale'}), 200

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def index():
    return render_template('index.html')

original_img = None  # Global variable to store the original image

@app.route('/upload', methods=['POST'])
def upload():
    global processed_img, original_img, undo_stack, redo_stack
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if file:
        # Process the uploaded file
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        original_img = img.copy()  # Store the original image
        processed_img = img.copy()
        undo_stack = [processed_img.copy()]  # Initialize undo stack
        redo_stack = []  # Clear redo stack
        return jsonify({'success': True, 'message': 'Image uploaded successfully'}), 200
    return jsonify({'error': 'Invalid file.'}), 400

@app.route('/get_original_image')
def get_original_image():
    global original_img
    if original_img is not None:
        _, buffer = cv2.imencode('.jpg', original_img)
        encoded_image = buffer.tobytes()
        return send_file(BytesIO(encoded_image), mimetype='image/jpeg')
    return jsonify({'error': 'No original image to display.'}), 400

@app.route('/rotate', methods=['POST'])
def rotate_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        angle = int(request.form.get('angle', 90))  # Get desired angle from user
        (h, w) = processed_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(processed_img, M, (w, h))
        processed_img = rotated
        undo_stack.append(processed_img.copy())  # Save state for undo
        redo_stack.clear()  # Clear redo stack
    return jsonify({'success': True, 'message': f'Image rotated by {angle} degrees'}), 200

@app.route('/invert', methods=['POST'])
def invert_colors():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        processed_img = cv2.bitwise_not(processed_img)
        undo_stack.append(processed_img.copy())  # Save state for undo
        redo_stack.clear()  # Clear redo stack
    return jsonify({'success': True, 'message': 'Colors inverted'}), 200

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
    return jsonify({'success': True, 'message': f'Image cropped to {width}x{height}'}), 200

@app.route('/undo', methods=['POST'])
def undo():
    global processed_img, undo_stack, redo_stack
    if len(undo_stack) > 1:
        redo_stack.append(undo_stack.pop())  # Move current state to redo stack
        processed_img = undo_stack[-1].copy()  # Revert to the last state
    return jsonify({'success': True, 'message': 'Undo performed'}), 200

@app.route('/redo', methods=['POST'])
def redo():
    global processed_img, undo_stack, redo_stack
    if redo_stack:
        undo_stack.append(redo_stack.pop())  # Move last redo state to undo stack
        processed_img = undo_stack[-1].copy()  # Set to the redo state
    return jsonify({'success': True, 'message': 'Redo performed'}), 200

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

@app.route('/download')
def download_image():
    global processed_img
    if processed_img is not None:
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg', as_attachment=True, download_name='edited_image.jpg')
    return jsonify({'error': 'No image to download.'}), 400

@app.route('/get_image')
def get_image():
    global processed_img
    if processed_img is not None:
        _, buffer = cv2.imencode('.jpg', processed_img)
        encoded_image = buffer.tobytes()
        return send_file(BytesIO(encoded_image), mimetype='image/jpeg')
    return jsonify({'error': 'No image to display.'}), 400

# Filters

@app.route('/sharpen', methods=['POST'])
def sharpen_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
        processed_img = cv2.filter2D(processed_img, -1, kernel)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image sharpened'}), 200

@app.route('/saturate', methods=['POST'])
def saturate_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        
        # Get the saturation factor from the user input (default is 1.5)
        saturation_factor = float(request.form.get('factor', 1.5))
        
        # Scale the saturation channel
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        
        # Convert back to BGR color space
        processed_img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        # Save state for undo
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    
    return jsonify({'success': True, 'message': f'Image saturation adjusted by factor {saturation_factor}'}), 200

@app.route('/smooth', methods=['POST'])
def smooth_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        processed_img = cv2.GaussianBlur(processed_img, (15, 15), 0)  # Smoothing with Gaussian Blur
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image smoothed'}), 200

@app.route('/edge_detect', methods=['POST'])
def edge_detect():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        processed_img = cv2.Canny(processed_img, 100, 200)  # Edge detection using Canny
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for consistent display
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Edge detection applied'}), 200

@app.route('/emboss', methods=['POST'])
def emboss_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        kernel = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])  # Emboss kernel
        processed_img = cv2.filter2D(processed_img, -1, kernel) + 128  # Add 128 to make effect visible
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Emboss effect applied'}), 200

@app.route('/color_balance', methods=['POST'])
def color_balance():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        result = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
        l, a, b = cv2.split(result)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE for contrast enhancement
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # Convert back to BGR color space
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Color balance adjusted'}), 200

@app.route('/segment', methods=['POST'])
def segment_image():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert to grayscale
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove noise using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the unknown region with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(processed_img, markers)
        processed_img[markers == -1] = [255, 0, 0]  # Mark boundaries in red

        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    
    return jsonify({'success': True, 'message': 'Region-based segmentation applied'}), 200

@app.route('/tonal_transformation', methods=['POST'])
def tonal_transformation():
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        tonal_type = request.form.get('tonal_type', 'brightness_contrast')

        if tonal_type == 'brightness_contrast':
            # Adjust brightness and contrast
            alpha = 1.5  # Contrast control
            beta = 50    # Brightness control
            processed_img = cv2.convertScaleAbs(processed_img, alpha=alpha, beta=beta)

        elif tonal_type == 'highlights_shadows':
            # Adjust highlights and shadows
            hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            hsv[:, :, 2] = cv2.equalizeHist(v)
            processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif tonal_type == 'levels_curves':
            # Adjust levels and curves
            lut = np.zeros((256,), dtype=np.uint8)
            for i in range(256):
                lut[i] = np.clip(128 + (i - 128) * 1.5, 0, 255)
            processed_img = cv2.LUT(processed_img, lut)

        undo_stack.append(processed_img.copy())
        redo_stack.clear()

    return jsonify({'success': True, 'message': f'{tonal_type} transformation applied'}), 200

@app.route('/segmentation')
def segmentation_page():
    return render_template('segmentation.html')

@app.route('/perform_segmentation', methods=['POST'])
def perform_segmentation():
    global model, device, preprocess
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        # Read and decode the uploaded image
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert to PIL Image
        input_image = Image.fromarray(img_rgb)

        # Preprocess the image
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Define vehicle classes based on COCO dataset class IDs
        vehicle_classes = [2, 5, 7]  # Car, Bus, Truck

        # Create a mask for vehicles
        vehicle_mask = np.isin(output_predictions, vehicle_classes).astype(np.uint8) * 255

        # Resize the mask to match the original image size
        vehicle_mask = cv2.resize(vehicle_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a color mask
        color_mask = np.zeros_like(img)
        color_mask[vehicle_mask == 255] = [0, 0, 255]  # Red color for vehicles

        # Blend the color mask with the original image
        alpha = 0.5
        blended_img = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

        # Encode the blended image to send back
        _, buffer = cv2.imencode('.jpg', blended_img)
        encoded_image = buffer.tobytes()

        # Return the processed image as response
        return send_file(BytesIO(encoded_image), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
