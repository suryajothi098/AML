import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

print("Test accuracy:", model.evaluate(x_test, y_test)[1])

for a, b, title in [('accuracy', 'val_accuracy', 'CNN Model Accuracy'),
                    ('loss', 'val_loss', 'CNN Model Loss')]:
    plt.plot(h.history[a], label='Training ' + a.title())
    plt.plot(h.history[b], label='Validation ' + a.title())
    plt.title(title)
    plt.legend()
    plt.show()
