import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random


def load_dataset(N=30000,NP=1800):

	obstacles = np.zeros((N,4002),dtype=np.float32)
	for i in range(0, N):
		temp = np.fromfile('../../data_generation/obs.dat')
		temp = temp.reshape(int(len(temp)/2),2)
		print(temp.shape)
		obstacles[i] = temp.flatten()

	return obstacles


if __name__ == '__main__':
	load_dataset()
