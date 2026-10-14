import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)



x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)


model = models.Sequential([layers.Embedding(10000, 128),layers.LSTM(128),layers.Dense(1, activation='sigmoid')])


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


h = model.fit(x_train, y_train,epochs=3,batch_size=32,validation_split=0.1)


loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

plt.plot(h.history['accuracy'], label='Training Accuracy')
plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.plot(h.history['loss'], label='Training Loss')
plt.plot(h.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
