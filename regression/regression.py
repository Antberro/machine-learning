import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

device = 'cpu'


# 1) Generate Data
np.random.seed(50)
x = np.random.rand(100, 1)
x = np.sort(x, axis=0)
y = 1 + -4*x + np.random.rand(100, 1)  

# convert numpy data into tensors
x_tensor = torch.from_numpy(x).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

# create dataset
dataset = TensorDataset(x_tensor, y_tensor)

# split into training and validation data
train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)


# 2) Create Model
torch.manual_seed(50)
model = nn.Sequential(
	nn.Linear(1, 1)).to(device)

# set learning rate
lr = 0.1

# set num of epochs
num_epochs = 500

# define loss function
loss_fn = nn.MSELoss(reduction='mean')


# define optimizer to update parameters
optimizer = optim.SGD(model.parameters(), lr=lr)


def generate_training_step(model, loss_fn, optimizer):
	# returns a function that goes through one loop of training
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


# 3) Train Mode
training_step = generate_training_step(model, loss_fn, optimizer)
losses = []
val_losses = []

for epoch in range(num_epochs):
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


# 4) Display Results
print(model.state_dict())

# plot model
plt.scatter(x, y, color='blue')
p = model(x_tensor).detach().numpy()
plt.plot(x, p, color='red')
plt.show()

# plot loss
t1 = [i for i in range(len(losses))]
t2 = [i for i in range(len(val_losses))]
plt.plot(t1, losses, color='red')
plt.plot(t2, val_losses, color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

