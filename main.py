import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os

IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 16  # Número de imagens processadas por vez
EPOCHS = 10  # Número de vezes que o modelo verá os dados
DATASET_PATH = "dataset"  # Pasta principal do dataset

# Carregando e Pré-processando as Imagens
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normaliza e separa 20% para validação

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Criando a CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Evita overfitting
    Dense(3, activation='softmax')  # Três saídas
])

# Compilando e Treinando a CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

# Salvando o modelo treinado
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/model_v1.h5")
