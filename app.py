from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

model_path = ['model0.h5', 'model1.h5', 'model2.h5']
models = [tf.keras.models.load_model(path) for path in model_path]


@app.post('/predict')
async def post_picture(picture: UploadFile = File(...)):
    try:
        image = Image.open(picture.file).convert("L")
        processed_image = process_image(image)

        prediction = [model.predict(processed_image) for model in models]
        prediction_digit = [int(np.argmax(pred)) for pred in prediction]

        return {'digit': prediction_digit}

    except Exception as e:
        return {'error': str(e)}


def process_image(image_array):
    image_array = np.array(image_array)

    image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
    _, image_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_array = cv2.resize(image_array, (28, 28))
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array
