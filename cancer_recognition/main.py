from cancer_recognition.convolutional_neural_network import create_and_train_convolutional_neural_network
from cancer_recognition.plot_images import plot_images
from cancer_recognition.resize_images import resize_all
from cancer_recognition.neural_network import create_and_train_network, format_data

resize_all()
#
# (train_images, train_labels, test_images, test_labels) = format_data()
# model = create_and_train_network()
model = create_and_train_convolutional_neural_network()
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)

# predictions = model.predict(test_images)
# print(predictions)
# print(test_labels)
# plot_images(predictions, test_labels, test_images)
