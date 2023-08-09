import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


# DMLP Model-Path Generator 
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		multiple = 1
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280*multiple),nn.PReLU(),nn.Dropout(),
		nn.Linear(1280*multiple, 1024*multiple),nn.PReLU(),nn.Dropout(),
		nn.Linear(1024*multiple, 896*multiple),nn.PReLU(),nn.Dropout(),
		nn.Linear(896*multiple, 768*multiple),nn.PReLU(),nn.Dropout(),
		nn.Linear(768*multiple, 512*multiple),nn.PReLU(),nn.Dropout(),
		nn.Linear(512*multiple, 384*multiple),nn.PReLU(),nn.Dropout(),
		nn.Linear(384*multiple, 256*multiple),nn.PReLU(), nn.Dropout(),
		nn.Linear(256*multiple, 256*multiple),nn.PReLU(), nn.Dropout(),
		nn.Linear(256*multiple, 128*multiple),nn.PReLU(), nn.Dropout(),
		nn.Linear(128*multiple, 64*multiple),nn.PReLU(), nn.Dropout(),
		nn.Linear(64*multiple, 32*multiple),nn.PReLU(),
		nn.Linear(32*multiple, output_size))

	def forward(self, x):
		out = self.fc(x)
		# print('1',out.shape)
		# out = torch.tensor([(o if o > 0.1**10 else 0) for o in out]).cuda()
		# if out[0] < 0.1**10:
		# 	out[0]=0
		# if out[1] < 0.1**10:
		# 	x[1]=0
		# print('2', x)
		return torch.tensor([(o if o > 0.1**10 else 0) for o in out]).cuda()

 
