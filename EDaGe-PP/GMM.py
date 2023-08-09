import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import distributions


class GMM:
    def __init__(self, order=10, dim=2, mean_range=70, std_range=5):
        self.Order = order
        self.Dim = dim
        mean = torch.rand([self.Order, self.Dim])*mean_range
        std = torch.rand([self.Order, self.Dim])*std_range
        weights = torch.rand([self.Order])
        mix = distributions.Categorical(weights)
        comp = distributions.Independent(distributions.Normal(mean, std), 1)
        self.Distribution = distributions.MixtureSameFamily(mix, comp)


if __name__ == '__main__':
    model = GMM(10, 2)
    samp = model.Distribution.sample([10000, ])
    plt.plot(np.transpose(np.reshape(samp, [10000, 2]))[0], np.transpose(np.reshape(samp, [10000, 2]))[1], 'ko')
    plt.savefig(r'./map224.png', dpi=90)
    plt.show()
