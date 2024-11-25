import pandas as pd
import numpy as np
import struct
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
warnings.filterwarnings('ignore', category=FutureWarning)


def load_mnist_data(i_filepath, l_filepath):
    with open(i_filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError("The file is not a valid image file.")

        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)

    with open(l_filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError("The file is not a valid label file.")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    df = pd.DataFrame(images)
    df["label"] = labels

    print('Data after loading:')
    print(df.head())

    # Normalize the pixel values
    df.iloc[:, :-1] = df.iloc[:, :-1].astype('float32') / 255.0
    one_hot_labels = pd.get_dummies(df['label'], prefix='digit')

    df = pd.concat([df.iloc[:, :-1], one_hot_labels], axis=1)

    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    print('Data after normalization:')
    print(df.head())
    return df


def build_cnn():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='linear'),
    ],
        name='CNN1'
    )

    model1 = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(50, kernel_size=(4, 4), activation='relu'),
        Conv2D(100, kernel_size=(4, 4), activation='relu'),
        Conv2D(200, kernel_size=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='linear')
    ],
        name='CNN2'
    )

    model2 = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        Conv2D(128, kernel_size=(5, 5), activation='relu'),
        Conv2D(256, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='linear')

    ],
        name='CNN3'
    )

    return [model, model1, model2]
