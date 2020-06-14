import plotting.visualize as viz
from classes.neural_net import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim
from pathlib import Path
from sklearn.datasets import make_classification

def plot_binary_data(x, y, R=None):
	'''
	Function that plots binary classification data.

	Takes in Parameters:
		- x: Numpy Array; array of x values with shape (n,2) with n data points with 2 features (x1, x2)
		- y: Numpy Array; array of y values with shape (n,1) with n data points with 1 feature (1, 0)
		- R: Integer; (optional) bound for size of plot, will draw axes from (-R,R) 
	'''
	n, _ = x.shape
	fig, ax = viz.initialize_figax(title='Data', xlabel='X1', ylabel='X2')
	for i in range(n):
		mcolor = None
		if y[i,0] == 1: mcolor = viz.GREEN
		elif y[i,0] == 0: mcolor = viz.RED
		ax.scatter([x[i,0]], [x[i,1]], color=mcolor)
	if R:
		plt.xlim(-R, R)
		plt.ylim(-R, R)
	plt.show()


# GENERATE DATA
# dataset 1
x1, y1 = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=9990)
y1 = np.reshape(y1, (100, 1))
if False:
	plot_binary_data(x1, y1, R=5)

# dataset 2
x2, y2 = x, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=999)
y2 = np.reshape(y2, (100, 1))
if False:
	plot_binary_data(x2, y2, R=5)

# xor dataset
x3 = np.array([[-1, 1], [1, -1], [1, 1], [-1, -1]])
y3 = np.array([[1, 1, 0, 0]]).T
if False:
	plot_binary_data(x3, y3, R=5)


# STEP 2) CREATE MODEL
arch1 = nn.Sequential(
	nn.Linear(2, 5),
	nn.Sigmoid(),
	nn.Linear(5,2))

if True:
	# set hyperparams/loss/optimizer
	net = NeuralNet(arch1)
	net.set_hyper_params(lr=1e-1, batch_size=1, epochs=500)
	net.set_data(x1, y1)
	net.set_loss_fn(nn.CrossEntropyLoss())
	net.set_optimizer(optim.SGD)


# STEP 3) TRAIN MODEL AND VIEW SUMMARY
if True:
	net.learn()
	net.plot_loss()