from cancer_recognition.convolutional_neural_network import create_and_train_convolutional_neural_network
from cancer_recognition.plots import plot_training_progress, plot_model_to_file
from cancer_recognition.resize_images import resize_all

width, height = resize_all()
print(width, height)

model, history = create_and_train_convolutional_neural_network(width, height)
plot_training_progress(history)
plot_model_to_file(model)
