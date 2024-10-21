from flask import Flask, request, render_template, redirect, url_for, flash
import os
import time
import numpy as np
import PIL.Image
import tensorflow as tf
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.secret_key = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

model_path = './model'
loaded_model = tf.saved_model.load(model_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/style_transfer', methods=['GET', 'POST'])
def upload_image():
    content_image_path = None
    style_image_path = None
    stylized_image_path = None

    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file = request.files['style_image']

        if content_file and style_file:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))
            content_file.save(content_path)
            style_file.save(style_path)

            content_image = load_img(content_path)
            style_image = load_img(style_path)
            stylized_image = loaded_model(tf.constant(content_image), tf.constant(style_image))[0]
            result_image = tensor_to_image(stylized_image)

            stylized_image_path = 'result.png'  # Path for the stylized image
            result_image.save(os.path.join('static', stylized_image_path))

            # Set the paths for the uploaded images
            content_image_path = secure_filename(content_file.filename)
            style_image_path = secure_filename(style_file.filename)

    return render_template('upload.html', stylized_image=stylized_image_path, 
                           content_image=content_image_path, style_image=style_image_path)

 

if __name__ == '__main__':
    app.run(debug=True, port=5001)  

