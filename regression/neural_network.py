import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import numpy as np 

class NeuralNet:
	def __init__(self, model):
		self.device = 'cpu'
		self.model = model

		self.raw_x, self.raw_y = None, None
		self.x_tensor, self.y_tensor = None, None
		self.train_loader, self.val_loader = None, None

		self.lr = None
		self.batch_size = None
		self.train_val_split = None
		self.num_epochs = None

		self.loss_fn = None
		self.optimizer = None

		self.train_losses = []
		self.val_losses = []

	# private methods
	def _generate_train_step(self):
		'''
		Internal method used to generate a function that trains the model over a single epoch.
		'''
		def _train_step(x, y):
			# enter training mode
			self.model.train()
			# compute predictions and loss
			pred = self.model(x)
			loss = self.loss_fn(y, pred)
			loss.backward()
			# update parameters and reset gradients
			self.optimizer.step()
			self.optimizer.zero_grad()
			# return loss
			return loss.item()
		return _train_step


	# public methods
	# TODO: make splits ratio more general so it works with any n != 100
	def set_hyper_params(self, **kwargs):
		'''
		Sets hyperparameters for learning algorithm.

		Takes in keyword arguments:
			- lr: Int; learning rate
			- epochs: Int; number of epochs for training
			- batch_size: Int; number of data points in mini-batch (1 for SGD)
			- split: List[Int,Int]; train/val split of data (default of 80/20 for 100 data points)
		'''
		self.lr = kwargs['lr'] if 'lr' in kwargs else None
		self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
		self.train_val_split = kwargs['split'] if 'split' in kwargs else [80, 20]
		self.num_epochs = kwargs['epochs'] if 'epochs' in kwargs else None


	def set_data(self, x, y):
		'''
		Insert data into model to be split into training and validation data.
		Also saves numpy and tensor versions of data.
		
		Takes in parameters:
			- x: numpy array; array of x values with shape (n,d) with n data points with d features
			- y: numpy array; array of y values with shape (n,d) with n data points with d features
		'''
		try:
			# save numpy version for easy plotting
			self.raw_x = x
			self.raw_y = y
			# convert numpy data into tensors
			self.x_tensor = torch.from_numpy(x).float().to(self.device)
			self.y_tensor = torch.from_numpy(y).float().to(self.device)
			dataset = TensorDataset(self.x_tensor, self.y_tensor)
			# split into train/val datasets
			train_dataset, val_dataset = random_split(dataset, self.train_val_split)
			self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)
			self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.train_val_split[1])

		except TypeError:
			print('Need to call NeuralNet.set_hyper_params() before setting data!')


	def set_loss_fn(self, loss_func):
		'''
		Sets loss function for training.

		Takes in parameter:
			- loss_func: pytorch loss function object
		'''
		self.loss_fn = loss_func

	
	def set_optimizer(self, optimizer):
		'''
		Sets optimizer for updating parameters during training.
		
		Takes in parameter: pytorch optimizer object
		'''
		try:
			self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
		except TypeError:
			print('Need to call NeuralNet.set_hyper_params() before setting optimizer!')


	def learn(self):
		'''
		Trains model with training data.
		'''
		train_step = self._generate_train_step()
		for _ in range(self.num_epochs):
			for x_batch, y_batch in self.train_loader:
				x_batch = x_batch.to(self.device)
				y_batch = y_batch.to(self.device)

				loss = train_step(x_batch, y_batch)
				self.train_losses.append(loss)
			# validation
			with torch.no_grad():
				for x_val, y_val in self.val_loader:
					x_val = x_val.to(self.device)
					y_val = y_val.to(self.device)

					# enter eval mode
					self.model.eval()

					pred = self.model(x_val)
					val_loss = self.loss_fn(y_val, pred)
					self.val_losses.append(val_loss.item())

	def save_model(self):
		'''save model'''
		pass

	def predict(self, test_data):
		'''use model to predict new data'''
		pass

	def plot_model(self):
		'''plot data along with model's predictions'''
		pass

	def plot_loss(self):
		'''plot loss'''
		pass

if __name__ == "__main__":

	xdata = np.linspace(0, 10, 100).reshape(100, 1)
	ydata = -2 + 3*np.sin(xdata) + np.random.rand(100, 1)

	net = NeuralNet(nn.Sequential(nn.Linear(1, 1)))
	net.set_hyper_params(lr=1e-2, batch_size=1, epochs=1000)
	net.set_data(xdata, ydata)
	net.set_loss_fn(nn.MSELoss(reduction='mean'))
	net.set_optimizer(optim.SGD)

	