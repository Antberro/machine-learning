import helpers.visualize as viz
from helpers.neural_net import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim

# STEP 1) GENERATE DATA TO FIT MODEL
x = np.linspace(0, 10, 100).reshape(100, 1)
y = -4 + 3*np.cos(x) + np.random.rand(100, 1)

if False:  # plot data
	fig, ax = viz.initialize_figax(title='Data')
	ax.scatter(x, y, color=viz.BLUE)
	plt.show()


# STEP 2) CREATE MODEL
arch = nn.Sequential(  # select network architechture
	nn.Linear(1, 50),
	nn.ReLU(),
	nn.Linear(50, 50),
	nn.ReLU(),
	nn.Linear(50, 1))

# set hyperparams/loss/optimizer
net = NeuralNet(arch)
net.set_hyper_params(lr=1e-3, batch_size=1, epochs=500)
net.set_data(x, y)
net.set_loss_fn(nn.MSELoss(reduction='mean'))
net.set_optimizer(optim.SGD)


# STEP 3) TRAIN MODEL AND VIEW SUMMARY
net.learn()
net.plot_model()
net.plot_loss()