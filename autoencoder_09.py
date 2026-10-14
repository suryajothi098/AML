
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)


model = models.Sequential([layers.Input(shape=(784,)),layers.Dense(128, activation='relu'),layers.Dense(64, activation='relu'),layers.Dense(128, activation='relu'),layers.Dense(784, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy')


h = model.fit(x_train, x_train,epochs=3,batch_size=32,validation_data=(x_test, x_test))


decoded_imgs = model.predict(x_test)

plt.figure(figsize=(10, 4))

for i in range(10):
  
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()



