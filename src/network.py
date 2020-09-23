import numpy as np 
import random

def sigmoid(X):
	return 1 / (1 + np.exp(X))

def sigmoid_prime(X):
	return sigmoid(X) * (1 - sigmoid(X))

def identity(X):
	return X

def identity_prime(X):
	return 1

class Network():
	"""
	Creates a new NN with specified layers or loads a network from a specified path 
	Throws assertion error if file_name is not specified 
	If layer_list is provided, creates a new network, if not loads network from path
	layer_list must be an iterable with (int, (activationFunction, activationFuntionPrime))
	"""
	def __init__(self, layer_list):
		# we store the network as a linked list, 
		# and we will preform updates/edits to the 
		# network recursively 
		self.last_layer = None
		self.input_size, self.output_size = layer_list[0], layer_list[-1]

		prev = None
		for i in range(len(layer_list)-1):
			# TODO MAKE ACTIVATION A THING
			temp = Layer(layer_list[i], layer_list[i + 1], prev_layer = prev)
			prev = temp

		self.last_layer = temp 


	"""
	Forward propagates the input of the network
	"""
	def __call__(self, X):
		return self.last_layer(X)


	"""
	trains the network with data provided 
	for the number of epochs or until the network has error less than the 
	threshold 
	"""
	def train(self, X_train, Y_train, batch_size=10, num_epochs=1000):
		# Catching the bugs
		## TODO MAKE X_train able to be a generator function :)s 
		assert len(X_train) == len(Y_train), "Training data not matched"

		if len(X_train) < batch_size:
			batch_size = len(X_train)

		# setting up our loop variable s
		train_index = list(range(len(X_train)))
		i = 0 
		while i < num_epochs:
			i += 1 
			self.last_layer.start_batch()
			# we use train index so we don't have to worry 
			# about order being lost with random.sample
			batch = random.sample(train_index, batch_size)
			batch_error = np.zeros(self.output_size)
			for j in batch:
				Y = self.last_layer(X_train[j]) 
				batch_error = batch_error - (Y - Y_train[j])

			self.last_layer.back_propagate(batch_error)


	"""
	Save the network to the specified filename
	If none is provided, defaults to the one specified
	in initialization
	"""

	def __str__(self):
		return self.last_layer.__str__()

class Layer():

	"""
	Creates a new layer of neurons
	x_size, y_size correspond to the size of the input, output vectors, respectively 
	prev_layer is the previous layer in the network if None, this is the input layer
	next_layer is the next layer in the network if None, this is the output layer 
	activation is the type of activation this layer should use 
	passed as [f:R->R, f']
	"""
	def __init__(self, x_size, y_size, prev_layer=None, next_layer=None, activation_function=(sigmoid, sigmoid_prime)):
		self.W = np.random.rand(x_size, y_size)
		self.B = np.random.rand(y_size)
		self.prev_layer = prev_layer
		self.next_layer = next_layer
		self.activation = activation_function[0]
		self.activation_prime = activation_function[1] 
		self.training = False
		self.num_examples = 0 
		self.ave_output = np.zeros(y_size, dtype=np.float64)

	"""
	Back propagates the aggregate error
	since start_batch was called 
 	"""
	def back_propagate(self, error, learning_rate=0.001):
		self.training = False

		error = error * self.activation_prime(self.ave_output)

		dW = learning_rate * self.ave_output * error
		dB = learning_rate * error 

		self.W = self.W - dW
		self.B = self.B - dB
 		
		if self.prev_layer:
			### RECURSION 
			error = np.dot(error, self.W.T)
			self.prev_layer.back_propagate(error, learning_rate=learning_rate)


	def __call__(self, X):
		## RECURSION :) 
		if self.prev_layer:
			X = self.prev_layer(X)

		Z = self.activation(np.dot(X, self.W) + self.B)

		##Keeps track of the output of the layer for training 
		if self.training:
			self.num_examples += 1
			self.ave_output = self.ave_output + Z
			if self.num_examples > 1: 
				self.ave_output *= (self.num_examples - 1) / self.num_examples

		return Z 

	"""
	Starts a training batch 
	"""
	def start_batch(self):
		self.training = True
		self.num_examples = 0 
		self.ave_output = np.zeros(len(self.B), dtype=np.float64)
		if self.prev_layer:
			self.prev_layer.start_batch()

	def __str__(self):
		s = ""
		if self.prev_layer:
			s = self.prev_layer.__str__()
		return s + "\n W =" + self.W.__str__() + "\n B =" + self.B.__str__() 








