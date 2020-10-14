import numpy as np
import pandas as pd

df = pd.read_csv('/Users/agnezainyte/PycharmProject/F21BC/diabetes_2.csv')

X = np.array(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
Y = np.array(df[['Outcome']])

print("\nHere is a brief description about data set...")
print(df.head(11))
print("The description of the dataset about diabetes.\nFirst number describes instances and second attributes\n", X.shape)

m = len(X)
print("The number of training samples is: ", m)
w = np.ones_like(X[0])
b = 0

# CORRECT
def sigmoid(z):
	return 1/(1 + np.math.exp(-z))

def gradient_descent(X, Y, w, b, m, alpha):
	print("\nLearning rate has been initialised to ", alpha)
	dw = np.zeros_like(X)
	db = 0
	z = np.zeros(len(X))
	print(z.shape)
	a = np.zeros(len(X))
	L = 0
	for i in range(1, m):
		z[i] = (np.dot(np.transpose(w), X[i]) + b)
		a[i] = sigmoid(z[i])
		np.seterr(divide='ignore')
		L += (-(Y[i] * np.log(a[i]) + (1 - Y[i]) * np.log(1-a)))
		dz = a[i] - Y[i]
		dw[i] = X[i] * dz
		db += dz
	# dw /=m
	db /=m
	print("Here is how weights were adjusted during training: ")
	for i in range(1,len(w)):
		mean_dw = np.sum(dw[i])/8
		w[i] = w[i] - alpha * mean_dw
		print("Attributes number and its weight ", i, w[i])
	b = b - alpha * db
	return L


J = gradient_descent(X, Y, w, b, m, 2)
print(J)


