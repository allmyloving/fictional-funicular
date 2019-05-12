from cancer_recognition.plot_images import plot_images
from cancer_recognition.resize_images import resize_all
from cancer_recognition.neural_network import create_and_train_network, format_data

# resize_all()

(train_images, train_labels, test_images, test_labels) = format_data()
model = create_and_train_network(train_images, train_labels)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions)
print(test_labels)
plot_images(predictions, test_labels, test_images)
