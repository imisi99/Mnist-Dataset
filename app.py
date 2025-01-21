from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np

app = FastAPI()

model_path = ['model0.h5', 'model1.h5', 'model2.h5']
models = [tf.keras.models.load_model(path) for path in model_path]


@app.post('/predict')
async def post_picture(picture: UploadFile = File(...)):
    try:
        image = Image.open(picture.file).convert("L")
        image = image.resize((28, 28))

        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        prediction = [model.predict(image_array) for model in models]
        prediction_digit = [np.argmax(pred) for pred in prediction]

        return {'digit': prediction_digit}

    except Exception as e:
        return {'error': str(e)}
