import neuralnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

# specify the layers, first layer should be 784 and output layer should be 10 for this example
layer_sizes = (784, 16, 16, 10)

# number of images used for training, the rest is used for testing the performance
training_set_size = 5000

training_set_images = training_images[:training_set_size]
training_set_labels = training_labels[:training_set_size]

test_set_images = training_images[training_set_size:]
test_set_labels = training_labels[training_set_size:]


# initialize neural network
net = nn.NeuralNetwork(layer_sizes)

# evaluate performance without training
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

# first training session
net.train_sgd(training_set_images, training_set_labels, 2, 10, 4.0)

# evaluate performance after first training session
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

# second training session
net.train_sgd(training_set_images, training_set_labels, 8, 20, 2.0)

# evaluate performance after second training session
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)
