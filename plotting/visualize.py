import matplotlib.pyplot as plt 

# colors
BLUE = '#025df0'
RED = '#f23333'
GREEN = '#47c94d'
PURPLE = '#8f4eb5'
ORANGE = '#ff854d'
GRAY = '#e1e4e8'
DARK = '#3b3b3b'

def initialize_figax(**kwargs):
	'''
	Function that either returns a stylized set of fig, ax objects to plot data.
	Or, if an existing Axis object is passed in, modifies it to the specific style.

	Takes in parameters:
		- ax(optional): Axis Object; axis to be modified
		- title(optional): String; title to be displayed above plot (default none)
		- xlabel(otional): String; text to be displayed below x-axus (default none)
		- ylabel(optional): String; text to be didsplayed along y-axis (default none)

	Returns:
		- If no Axis is passed in, returns tuple (fig, ax)
		- If Axis is passed in, returns a single modified ax
	'''
	# unpack arguments
	mcolor = kwargs['color'] if 'color' in kwargs else BLUE
	pxlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
	pylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''
	ptitle = kwargs['title'] if 'title' in kwargs else ''
	# create fig, ax if needed
	if 'ax' not in kwargs:
		fig, ax = plt.subplots(figsize=(7, 5))
	else:
		ax = kwargs['ax']
	# set background color
	ax.set_facecolor(GRAY)
	# turn gridlines white
	ax.grid(True, color='white')
	ax.set_axisbelow(True)
	# hide axis splines
	ax.spines["top"].set_visible(False)    
	ax.spines["bottom"].set_visible(False)    
	ax.spines["right"].set_visible(False)    
	ax.spines["left"].set_visible(False)
	# clear ticks
	ax.xaxis.set_tick_params(length=0)
	ax.yaxis.set_tick_params(length=0)
	# set labels
	ax.set_xlabel(pxlabel, labelpad=5, fontsize=12, color='black')
	ax.set_ylabel(pylabel, labelpad=10, fontsize=12, color='black')
	ax.set_title(ptitle, pad=10, fontsize=15, color='black')
	# return objects
	if 'ax' not in kwargs:
		return fig, ax
	else:
		return ax