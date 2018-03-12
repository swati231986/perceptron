import scipy.special
import numpy
class neuralNetwork:

	#initialization of the neural network
	def _init_(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.iNodes = inputnodes
		self.hNodes = hiddennodes
		self.oNodes = outputnodes
		self.lr = learningrate

		#weights from input layer to hidden later
		#self.wih = (numpy.random.rand(self.hNodes, self.iNodes) - 0.5)
		#mean is 0.0 standard deviation is pow(self.hNodes,-0.5)
		self.wih = numpy.random.normal(0.0, pow(self.hNodes,-0.5),(self.hNodes, self.iNodes))


		#weights from hidden layer to output layer
		#self.who = (numpy.random.rand(self.oNodes, self.hNodes) - 0.5)
		self.who = numpy.random.normal(0.0,pow(self.oNodes,-0.5), (self.oNodes, self.hNodes))

		#activation function- in our case it is a sigmoid function
		#lambda function
		self.activation_function = lambda x:scipy.special.expit(x)

		pass

	#train the neural network
	def train(self, input, target):

		#convert the input to 2D matrix
		input_list = numpy.array(input, ndmin = 2).T
		target_list = numpy.array(target, ndmin=2).T

		#calculate the output for the hidden layer
		hidden_inputs = numpy.dot(wih, input_list)
		hidden_outputs = self.activation_function(hidden_inputs)

		#calculate the final output
		final_inputs = numpy.dot(who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		#error is target- actual
		output_errors = target_list - final_outputs

		#errors of the hidden layer
		errors_hidden_layer = numpy.dot(self.who.T, output_errors)

		#final weights between hidden layer and output later
		self.who +=self.lr * numpy.dot((output_errors * final_outputs * (1-final_outputs)), numpy.transpose(final_outputs))

		#final weights between input layer and the hidden layer
		self.wih += self.lr* numpy.dot((errors_hidden_layer * hidden_outputs * (1-hidden_outputs)), numpy.transpose(hidden_outputs))
		pass

	#query the neural network
	def query(self, inputs):
		input_list = numpy.array(inputs, ndmin=2).T
		hidden_inputs = numpy.dot(self.wih, input_list)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		return final_outputs

	


# number of input, output and hidden nodes
inputNodes = 3
hiddenNodes = 3
outputNodes = 3


#learning rate
learningRate = 0.3

myNetwork = neuralNetwork()
myNetwork._init_(inputNodes, hiddenNodes, outputNodes, learningRate)