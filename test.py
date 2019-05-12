from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset_train = numpy.loadtxt("train.csv", delimiter=",")
dataset_test = numpy.loadtxt("test.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset_train[:, 0:8]
Y = dataset_train[:, 8]
X_test = dataset_test[:, 0:8]
Y_test = dataset_test[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
loss, accuracy= model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

probabilities = model.predict(X_test)
predictions = [float(round(x[0])) for x in probabilities]
accuracy = numpy.mean(predictions == Y_test)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

# predicted = model.predict(X)
# print([round(x[0]) for x in predicted])
# print(predicted)