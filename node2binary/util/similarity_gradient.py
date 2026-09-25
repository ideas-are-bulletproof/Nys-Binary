import torch

def sim_gradient(embeddings, samples, verbose=1):
    # embeddings: (n, d)
    # positive_samples: (?, 3)
    # negative_samples: (?, 3)
    gradient = torch.zeros(embeddings.size(), dtype=torch.int32, device=embeddings.device)

    d = embeddings.size(1)
    group_size = 10000

    # ok
    # for each row (a, b, m) we want to compute
    # 2 * dist(f(a), f(b)) * (1 - 2*xor(f(a), f(b))) * m
    # probably start by tiling the m, and see what happens

    for i in range(0, samples.size(0), group_size):
        allAs = torch.tile(samples[i: (i + group_size), :1], (1, d)).to(dtype=torch.int64)
        allBs = torch.tile(samples[i: (i + group_size), 1:2], (1, d)).to(dtype=torch.int64)
        allCounts = torch.tile(samples[i: (i + group_size), 2:], (1, d))
        # get the embeddings
        embA = torch.gather(embeddings, 0, allAs)
        embB = torch.gather(embeddings, 0, allBs)
        # if verbose >= 2:
        #    print("embA", embA)
        #    print("embB", embB)
        A_xor_B = torch.bitwise_xor(embA, embB)
        hamming_dist = A_xor_B.sum(dim=1, keepdim=True, dtype=torch.int32)
        gradient_pieces = (2 * hamming_dist * (1 - 2 * A_xor_B) + 1) * allCounts
        gradient.scatter_add_(0, allAs, gradient_pieces)
        gradient.scatter_add_(0, allBs, gradient_pieces)

        # reset the tensors
        for recycle in (allAs, allBs, allCounts, embA, embB, A_xor_B, hamming_dist, gradient_pieces):
            recycle.resize_(0)

    return gradient
