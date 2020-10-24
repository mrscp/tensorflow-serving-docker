import requests
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
img = test_images[:1]

payload = {"instances": img.tolist()}
# Make request
res = requests.post("http://localhost:8501/v1/models/mnist:predict", json=payload)
res = res.json()
print(res)
