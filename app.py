"""
Flask application for image processing with Basic Requirments of CROP , Colour Balance , Grey Scale etc and segmentation and Advanced Requirments
using a  DeepLabV3 trained by COCO dataset  model for DIP Group Project 

below is the Training Loop
Epoch Loop
for epoch in range(num_epochs):
    model.train()
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
├── masks/
    ├── image1.png
    ├── image2.png
"""

from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import torch
import torchvision
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)

# Load the  DeepLabV3 model for semantic segmentation again trained with CAR images from COCO dataset

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set the model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         std=[0.229, 0.224, 0.225]),
])

# Global variables to store the processed image and history for undo/redo
# Note: In a production environment, avoid using global variables for per-user data
# and consider implementing session management or a database to handle user states.
processed_img = None
original_img = None
undo_stack = []
redo_stack = []

@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('home.html')

@app.route('/app')
def index():
    """
    Render the main application page.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle image upload from the client.
    """
    global processed_img, original_img, undo_stack, redo_stack
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        # Read the uploaded image file as a NumPy array
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Store the original and processed images
        original_img = img.copy()
        processed_img = img.copy()

        # Initialize undo and redo stacks
        undo_stack = [processed_img.copy()]
        redo_stack = []

        return jsonify({'success': True, 'message': 'Image uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file.'}), 400

@app.route('/get_original_image')
def get_original_image():
    """
    Send the original uploaded image to the client.
    """
    global original_img
    if original_img is not None:
        _, buffer = cv2.imencode('.jpg', original_img)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')
    else:
        return jsonify({'error': 'No original image to display.'}), 400

@app.route('/get_image')
def get_image():
    """
    Send the current processed image to the client for display.
    """
    global processed_img
    if processed_img is not None:
        _, buffer = cv2.imencode('.jpg', processed_img)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')
    else:
        return jsonify({'error': 'No image to display.'}), 400

# Image Manipulation Routes

@app.route('/color', methods=['POST'])
def change_to_color():
    """
    Convert the processed image to RGB color space.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert image from BGR to RGB
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image converted to RGB color space'}), 200

@app.route('/black_white', methods=['POST'])
def change_to_black_white():
    """
    Convert the processed image to black and white.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert image to grayscale
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        _, bw_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Convert back to BGR format
        processed_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image converted to black and white'}), 200

@app.route('/grayscale', methods=['POST'])
def change_to_grayscale():
    """
    Convert the processed image to grayscale.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert image to grayscale and back to BGR for consistent format
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image converted to grayscale'}), 200

@app.route('/rotate', methods=['POST'])
def rotate_image():
    """
    Rotate the processed image by a specified angle.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        angle = int(request.form.get('angle', 90))  # Default angle is 90 degrees
        (h, w) = processed_img.shape[:2]
        center = (w // 2, h // 2)

        # Calculate rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated = cv2.warpAffine(processed_img, M, (w, h))

        processed_img = rotated
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': f'Image rotated by {angle} degrees'}), 200

@app.route('/invert', methods=['POST'])
def invert_colors():
    """
    Invert the colors of the processed image.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Invert colors
        processed_img = cv2.bitwise_not(processed_img)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Colors inverted'}), 200

@app.route('/crop', methods=['POST'])
def crop_image():
    """
    Crop the processed image to the specified width and height.
    """
    global processed_img, undo_stack, redo_stack
    width = int(request.form.get('width', 100))
    height = int(request.form.get('height', 100))
    if processed_img is not None:
        h, w = processed_img.shape[:2]
        start_x = max(w // 2 - width // 2, 0)
        start_y = max(h // 2 - height // 2, 0)
        # Ensure cropping area does not exceed image dimensions
        end_x = min(start_x + width, w)
        end_y = min(start_y + height, h)
        cropped_img = processed_img[start_y:end_y, start_x:end_x]
        processed_img = cropped_img
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': f'Image cropped to {width}x{height}'}), 200

# Undo and Redo Functionality

@app.route('/undo', methods=['POST'])
def undo():
    """
    Undo the last image processing action.
    """
    global processed_img, undo_stack, redo_stack
    if len(undo_stack) > 1:
        # Move the current state to redo stack
        redo_stack.append(undo_stack.pop())
        # Revert to the previous state
        processed_img = undo_stack[-1].copy()
    return jsonify({'success': True, 'message': 'Undo performed'}), 200

@app.route('/redo', methods=['POST'])
def redo():
    """
    Redo the last undone image processing action.
    """
    global processed_img, undo_stack, redo_stack
    if redo_stack:
        # Move the last redo state to undo stack
        undo_stack.append(redo_stack.pop())
        # Update the processed image
        processed_img = undo_stack[-1].copy()
    return jsonify({'success': True, 'message': 'Redo performed'}), 200

# Image Saving and Downloading

@app.route('/save', methods=['POST'])
def save_image():
    """
    Save the processed image and send it to the client for download.
    """
    global processed_img
    if processed_img is not None:
        # Convert processed image to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg', as_attachment=True, download_name='edited_image.jpg')
    else:
        return jsonify({'error': 'No image to save.'}), 400

@app.route('/download')
def download_image():
    """
    Send the processed image to the client for download.
    """
    global processed_img
    if processed_img is not None:
        # Convert processed image to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg', as_attachment=True, download_name='edited_image.jpg')
    else:
        return jsonify({'error': 'No image to download.'}), 400

# Image Filters

@app.route('/sharpen', methods=['POST'])
def sharpen_image():
    """
    Apply a sharpening filter to the processed image.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Define sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        # Apply filter
        processed_img = cv2.filter2D(processed_img, -1, kernel)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image sharpened'}), 200

@app.route('/saturate', methods=['POST'])
def saturate_image():
    """
    Adjust the saturation of the processed image.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)

        # Get the saturation factor from user input (default is 1.5)
        saturation_factor = float(request.form.get('factor', 1.5))

        # Adjust the saturation channel
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

        # Convert back to BGR color space
        processed_img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': f'Image saturation adjusted by factor {saturation_factor}'}), 200

@app.route('/smooth', methods=['POST'])
def smooth_image():
    """
    Apply a smoothing (Gaussian blur) filter to the processed image.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Apply Gaussian Blur
        processed_img = cv2.GaussianBlur(processed_img, (15, 15), 0)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Image smoothed'}), 200

@app.route('/edge_detect', methods=['POST'])
def edge_detect():
    """
    Apply edge detection to the processed image using the Canny algorithm.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Apply Canny edge detection
        edges = cv2.Canny(processed_img, 100, 200)
        # Convert edges to BGR format for consistent display
        processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Edge detection applied'}), 200

@app.route('/emboss', methods=['POST'])
def emboss_image():
    """
    Apply an emboss filter to the processed image.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Define emboss kernel
        kernel = np.array([[2, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        # Apply filter and add 128 to make the effect visible
        processed_img = cv2.filter2D(processed_img, -1, kernel) + 128
        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Emboss effect applied'}), 200

@app.route('/color_balance', methods=['POST'])
def color_balance():
    """
    Adjust the color balance of the processed image using CLAHE.
    """
    global processed_img, undo_stack, redo_stack
    if processed_img is not None:
        # Convert to LAB color space
        lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the channels back
        limg = cv2.merge((cl, a, b))

        # Convert back to BGR color space
        processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Color balance adjusted'}), 200

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Apply region-based segmentation to the processed image using the watershed algorithm.
    """
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

        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the unknown region with zero
        markers[unknown == 255] = 0

        # Apply watershed algorithm
        markers = cv2.watershed(processed_img, markers)
        processed_img[markers == -1] = [255, 0, 0]  # Mark boundaries in red

        undo_stack.append(processed_img.copy())
        redo_stack.clear()
    return jsonify({'success': True, 'message': 'Region-based segmentation applied'}), 200

@app.route('/tonal_transformation', methods=['POST'])
def tonal_transformation():
    """
    Apply tonal transformations to the processed image.
    """
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

        else:
            return jsonify({'error': f'Unknown tonal transformation type: {tonal_type}'}), 400

        undo_stack.append(processed_img.copy())
        redo_stack.clear()

    return jsonify({'success': True, 'message': f'{tonal_type} transformation applied'}), 200

# Segmentation Using Pre-trained Model

@app.route('/segmentation')
def segmentation_page():
    """
    Render the segmentation page.
    """
    return render_template('segmentation.html')

@app.route('/perform_segmentation', methods=['POST'])
def perform_segmentation():
    """
    Perform semantic segmentation on the uploaded image using the Trained COCO data set  DeepLabV3 model.
    """
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

        # Convert to PIL Image for processing
        input_image = Image.fromarray(img_rgb)

        # Preprocess the image
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Define vehicle classes based on COCO dataset class IDs
        vehicle_classes = [2, 5, 7]  # Car, Bus, Truck

        # Create a mask for vehicles
        vehicle_mask = np.isin(output_predictions, vehicle_classes).astype(np.uint8) * 255

        # Resize the mask to match the original image size
        vehicle_mask = cv2.resize(vehicle_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a color mask for visualization
        color_mask = np.zeros_like(img)
        color_mask[vehicle_mask == 255] = [0, 0, 255]  # Red color for vehicles

        # Blend the color mask with the original image
        alpha = 0.5
        blended_img = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

        # Encode the blended image to send back
        _, buffer = cv2.imencode('.jpg', blended_img)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
