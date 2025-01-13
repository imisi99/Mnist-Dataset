from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model('')
app.post('/predict')
async def post_picture(picture: UploadFile = File(...)):
    try:
        image = Image.open(picture.file).convert("L")
        image = image.resize((28, 28))

        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        prediction = model.predict(image_array)
        prediction_digit = np.argmax(prediction)

        return {'digit': prediction_digit}

    except Exception as e:
        return {'error': str(e)}