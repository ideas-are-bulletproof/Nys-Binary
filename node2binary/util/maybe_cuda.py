"""
maybe_cuda.py

Uses cuda if possible. Otherwise just uses the cpu.
"""

import torch

def _get_maybe_cuda():
    cuda = torch.device("cuda")
    if torch.cuda.is_available():
        try:
            t = torch.tensor([0], device=cuda)
            print(["CUDA enabled"][t.item()])
            return cuda
        except AssertionError:
            print("CUDA does not work")

    # we both seem to use Mac computers
    mps = torch.device("mps")
    if torch.backends.mps.is_available():
        #return mps # doesn't seem to work
        pass

    return torch.device("cpu")

maybe_cuda = _get_maybe_cuda()


