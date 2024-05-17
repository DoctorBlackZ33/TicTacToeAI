from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
# --- Definitions -------
def nn_model():
 model = Sequential()
 model.add(Dense(num_pixels, input_shape=(num_pixels,),
 kernel_initializer='normal', activation='sigmoid'))
 model.add(Dense(num_classes, kernel_initializer='normal',
 activation='softmax'))
 model.compile(loss='mean_squared_error', optimizer='SGD',
 metrics=['accuracy'])
 return model
# ---- main programm ------
# Datenset wird geladen. Trainings- und Testdaten werden aufgeteilt.
(trainX, trainY), (testX, testY) = mnist.load_data()
# Zusammenfassung des geladenen Datensets
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
# num_pixels ist Integerwert und gibt Anzahl der Pixels pro Bild an. 28*28=784 Pixels
num_pixels = trainX.shape[1] * trainX.shape[2]
# macht einen eindimensionalen Vektor aus jedem Bild
trainX = trainX.reshape((trainX.shape[0], num_pixels))
testX = testX.reshape((testX.shape[0], num_pixels))
# Werte normieren, damit sie besser verarbeitet werden können. All zwischen 0 und 1
trainX = trainX / 255
testX = testX / 255
# y-Werte one hot encoden
trainY = to_categorical(trainY)
testY = to_categorical(testY)
num_classes = testY.shape[1]
# Modell erstellen
model = nn_model()
# Modell Zusammenfassen (kann auskommentiert/weggelassen werden)
model.summary()
for layer in model.layers:
 print(layer.name)
 print(layer.get_weights()[0])
# Modell trainieren
history = model.fit(trainX, trainY, epochs=30, batch_size=300,
verbose=1)
model.save('model_trained_script.keras')
scores = model.evaluate(testX, testY, verbose=1)
print("Baseline Accuracy: %.2f%%" % (scores[1] * 100))
print("Baseline Loss: %.2f%%" % (scores[0] * 100))
# Modell evaluieren
figure, axis = plt.subplots(1, 2)
axis[0].plot(history.history['accuracy'])
axis[0].set_title('model accuracy')
axis[0].set_ylabel('accuracy')
axis[0].set_xlabel('epoch')
axis[0].legend(['train', 'test'], loc='upper left')
axis[1].plot(history.history['loss'])
axis[1].set_title('model loss')
axis[1].set_ylabel('loss')
axis[1].set_xlabel('epoch')
axis[1].legend(['train', 'test'], loc='upper left')
plt.show()