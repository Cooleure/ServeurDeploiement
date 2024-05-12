from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = tf.keras.models.load_model('./modelVGG19.keras')

IMAGE_SIZE = 300
CLASSES = ['TOEI', 'GB', 'WIT']


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Image</title>
    </head>
    <body>
        <h2>Upload Image</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """
    
@app.route('/predict', methods=['POST'])
def predict():

    image = request.files['image']
    img = Image.open(image)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img_array = np.asarray(img)
    reshaped_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(reshaped_array)
    classe_predite = str(CLASSES[np.argmax(prediction)])
    print(prediction)

    response = jsonify(classe_predite)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True)
