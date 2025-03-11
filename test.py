import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

IMG_WIDTH = 150
IMG_HEIGHT = 150

def test_image(image_path):
    model = tf.keras.models.load_model("models/model_v1.h5")  # Carrega o modelo treinado
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0  # Normaliza
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_names = ["Saudável", "Doença fúngica", "Seca"]
    print(f"Arquivo: {image_path} - Predição: {class_names[np.argmax(prediction)]}")


test_image("dataset/test/teste.jpg")
test_image("dataset/test/teste1.jpg")
test_image("dataset/test/teste2.jpg")