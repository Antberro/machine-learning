import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

device = 'cpu'

def package_data(x, y, batch_size):
	'''
	Takes in numpy arrays x, y of data along with a batch_size and splits
	into 80/20 training data and validation data. 

	Takes in parameters:
		- x: NUMPY ARRAY; (n,1) numpy array with n x-values
		- y: NUMPY ARRAY; (n,1) numpy array with n y-values
		- batch_size: INTEGER; number of data points in mini-batch (1 for SGD)
	
	Returns:
		- tuple (train_loader, val_loader) of pytorch DataLoader objects 
	'''
	# convert numpy data into tensors
	x_tensor = torch.from_numpy(x).float().to(device)
	y_tensor = torch.from_numpy(y).float().to(device)

	# create dataset
	dataset = TensorDataset(x_tensor, y_tensor)

	# split into training and validation data
	train_dataset, val_dataset = random_split(dataset, [80, 20])

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

	return train_loader, val_loader


def generate_training_step(model, loss_fn, optimizer):
	'''
	Used to generate a function that trains the model over a single epoch.

	Takes in parameters:
		- model: TORCH MODEL OBJ; model to undergo training
		- loss_fcn: FUNC; loss function to use for training
		- optimizer: TORCH OPTIM OBJ; optimizer to use for training
	
	Returns:
		- function train_step() that goes through one loop of training
	'''
	def train_step(x, y):
		# put model in training mode
		model.train()
		# compute predictions
		pred = model(x)
		# compute loss
		loss = loss_fn(y, pred)
		# compute gradients
		loss.backward()
		# update paramters and resets gradients
		optimizer.step()
		optimizer.zero_grad()
		# return loss
		return loss.item()
	return train_step


def fit_linear_regression(x, y, lr, num_epochs, batch_size, show=False):
	'''
	Creates a linear regression model and trains it to fit x,y data.

	Takes in parameters:
		- x: NUMPY ARRAY; (n,1) numpy array with n x-values
		- y: NUMPY ARRAY; (n,1) numpy array with n y-values
		- lr: FLOAT; learning rate for training
		- num_epochs: INTEGER; number of epochs to train model
		- batch_size: INTEGER; number of data points in mini-batch (1 for SGD)
		- show: BOOLEAN; True to display plot of data with prediction, False otherwise
	
	Returns:
		- trained model
	'''
	# put into dataloaders
	train_loader, val_loader = package_data(x, y, batch_size)

	# nn model for linear regression
	model = nn.Sequential(
		nn.Linear(1, 50),
		nn.ReLU(),
		nn.Linear(50, 50),
		nn.ReLU(),
		nn.Linear(50, 1)
		).to(device)

	# loss function for linear regression
	loss_fn = nn.MSELoss(reduction='mean')

	# define optimizer to update parameters
	optimizer = optim.SGD(model.parameters(), lr=lr)

	training_step = generate_training_step(model, loss_fn, optimizer)
	losses = []
	val_losses = []
	for _ in range(num_epochs):
		# train
		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			loss = training_step(x_batch, y_batch)
			losses.append(loss)

		# validation
		with torch.no_grad():
			for x_val, y_val in val_loader:
				x_val = x_val.to(device)
				y_val = y_val.to(device)

				# put model in eval mode
				model.eval()

				# compute val loss
				pred = model(x_val)
				val_loss = loss_fn(y_val, pred)
				val_losses.append(val_loss.item())

	# handle display options
	if show:
		# plot data
		plt.scatter(x, y, color='blue')
		# plot model
		x_tensor = torch.from_numpy(x).float()
		p = model(x_tensor).detach().numpy()
		plt.plot(x, p, color='red')
		plt.show()

	return model
	
	
if __name__ == '__main__':
	xdata = np.linspace(0, 10, 100).reshape(100, 1)
	ydata = -2 + 3*np.sin(xdata) + np.random.rand(100, 1) 
	model = fit_linear_regression(x=xdata, y=ydata, lr=1e-3, num_epochs=1000, batch_size=1, show=True)