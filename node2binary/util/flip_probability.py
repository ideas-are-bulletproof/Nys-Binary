import torch as torch
from .maybe_cuda import maybe_cuda

def softmax_prob(x):
    # convert to float
    x = x.float()

    # First we convert all the negative numbers to 0. Sample input x = [-3, 0, 1, 4, 8, -2] becomes [0, 0, 1, 4, 8, 0]
    x[x<0] = 0

    # Then we perform softmax for only the positions which are greater than 0.
    softmax_sum = torch.sum(torch.exp(x), keepdim=True)
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = torch.exp(x[i])/softmax_sum

    return x

def tanh_prob(x):
    return torch.maximum(torch.zeros(x.size(), device=maybe_cuda), torch.tanh(x.float() * 4.0) / 2.0)

def flip_prob(x):
    return tanh_prob(x)


if __name__ == "__main__":
    print(flip_prob(torch.tensor([3, 2, 0])))
    print(flip_prob(torch.tensor([0, 0, 0])))
    print(flip_prob(torch.tensor([-1, -2, -3, 0])))
    print(flip_prob(torch.tensor([-1, -2, -3, 4])))

