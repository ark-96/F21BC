import numpy as np
import pandas as pd

# Reading in input files using Pandas library
one_input_cubic_data = pd.read_csv("1in_cubic.txt", header=None, sep='\s+')
one_input_linear_data = pd.read_csv("1in_linear.txt", header=None, sep='\s+')
one_input_sine_data = pd.read_csv("1in_sine.txt", header=None, sep='\s+')
one_input_tanh_data = pd.read_csv("1in_tanh.txt", header=None, sep='\s+')
two_input_complex_data = pd.read_csv("2in_complex.txt", header=None, sep='\t')
two_input_xor_data = pd.read_csv("2in_xor.txt", header=None, sep='\t')

# Initialise random seed
np.random.seed(0)

# Activation functions
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def cosine(x):
	return np.cos(x)

def gaussian(x):
	return np.exp(-(x**2/2))


'''
def Neural_Network(self, X, num_neurons, num_layers):

Steps building a NN:
-identify how big is the input matrix
-create a weight matrix bu initialising random weights between (-1 and 1)
-create random bias
-for each num_neurons create neuron and populate NN
-for each num_layers populate hidden layers
-forward propagation

'''

def Neural_Network(input_file, num_neurons, num_layers):
	print("Initialising Neural Network...")
	print("Data used during the session is from {0}".format(input_file))
	X = input_file.iloc[:,:-1].values
	print("Check point 1.\nThe size of X matrix is {0} and it is type {1}".format(X.shape, X.dtype))
	print("Here are first 5 rows of input matrix X\n{0}".format(X[:5,:]))
	Y = input_file.iloc[:,-1:].values
	print("Check point 2.\nThe size of Y matrix is {0} and it is type {1}".format(Y.shape, Y.dtype))
	print("Here are first 5 rows of output matrix Y\n{0}".format(Y[:5, :]))

	m = len(X) #number of instances in the dataset
	n = len(X[0]) #number of attributes
	print("That is the breadth of input file is {0}".format(n))
	print("There are {0} instances in input file.".format(m))

	#creating weight matrix
	'''
	For each num_layers with num_neurons we need to intialise random weights
	'''
	w_input = np.random.rand(1, n)
	# w = np.random.rand(n, num_layers)
	print("Check point 3.\nHere is the weight matrix, all values were initialised randomly")
	print("The dimensions of w are:\n", w_input.shape)
	print("Here are first 5 rows of weight matrix\n{0}".format(w_input[:5,:]))

	#creating bias matrix
	b_input = np.random.randint(-2, 3, size = (1, n))
	# b = np.random.randint(-2, 3, size = (num_neurons, num_layers))
	print("Check point 4.\nHere is bias matrix, all values were initialised randomly")
	print("The dimensions of b are:\n", b_input.shape)
	print("Here are first 5 rows of bias matrix\n{0}".format(b_input[:5,:]))

	'''
	Creating activation function matrix, where it has:
	i number of rows corresponding to number of nodes
	j number of columns corresponding to number layers
	'''
	print("A matrix that will be holding an activation function  for input layer outputs is being created...")
	a1 = np.array([[0] * n] *1)
	z1 = np.array([[0] * n] *1)
	print("The shape of a1 matrix is {0}".format(a1.shape))
	print(a1[:5,:5])

	print("w_input shape {0}".format(w_input.shape))
	print("X shape {0}".format(X.shape))
	print("b_input shape: {0}".format(b_input.shape))

	z1 = (np.dot(X,np.transpose(w_input)))
	print("z1 shape before addition {0}".format(z1.shape))
	z1 = np.add(z1,b_input)
	print("z1 shape after addition {0}".format(z1.shape))
	a1 = sigmoid(z1)

	print("Check point 5.\nThis is how activation function matrix looks after passing through sigmoid: (first five lines)\n{0}".format(a1[:5,:]))
	print("This is the shape of a1 matrix {0}".format(a1.shape))

#write the code shitpile of bubonic baboon



	#foward propagation

	'''
	Here the code should populate first layer using one of the input activation functions
	Then send information further down the NN, where activation function might be ReLu
	???Iteration has to be column by column???
	'''

Neural_Network(two_input_complex_data, 10,2)








