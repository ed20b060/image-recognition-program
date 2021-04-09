# image-recognition-program
!pip install tensorflow==2.0-beta1
#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras
#Helper libraries
import numpy as np
import matplotlib as plt
print(tf.__version__)
class_names=["Tshirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
import matplotlib.pyplot as plt
def display_image(img):
  plt.imshow(img,cmap=plt.cm.binary)
  plt.show()

display_image(train_images[0]) 
train_images=train_images/255
test_images=test_images/255
model= keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics="accuracy"
)
model.fit(train_images,train_labels,epochs=10)
test_loss,test_accuracy=model.evaluate(test_images,test_labels)
print(test_accuracy)
predictions=model.predict(test_images)
for i in range(0,10):
  display_image(test_images[i])
  ind=np.argmax(predictions[i])
  print(class_names[ind])
