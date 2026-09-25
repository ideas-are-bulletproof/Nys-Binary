"""
evaluate_embedding.py

Given an embedding, evaluates it by determining how many of the positive and negative pairs are correctly classified.

Returns (TP, FN, FP, TN).
"""

import torch

def looks_like_is_a(emb1, emb2):
    return (
        torch.all(
            (torch.bitwise_and(emb1, emb2) == emb2) &
            (emb1 >= 0) & (emb2 >= 0),
        dim=-1, keepdim=True) &
        (torch.any(emb1 != emb2, dim=-1, keepdim=True))
    )

def evaluate_embedding_with_loss(embeddings, positive_is_a, negative_is_a):
    falseneg = torch.tensor([0], dtype=torch.int32, device=embeddings.device)
    falsepos = torch.tensor([0], dtype=torch.int32, device=embeddings.device)
    loss_pos = torch.tensor([0], dtype=torch.int32, device=embeddings.device)

    # This is very similar to sgd.py
    d = embeddings.size(1)

    # Process this in pseudo-minibatches to avoid excessive GPU memory usage
    # (this is NOT the minibatch parameter)
    group_size = 14000
    # reason:
    # 10000 pairs * 500 (maximum reasonable dimension) * 4 bytes per pos
    # = 20'000'000 bytes per array.
    # actually there are multiple copies but it's still well under 1GB
    for i in range(0, positive_is_a.size(0), group_size):
        allAs = positive_is_a[i:(i+group_size), :1].repeat(1, d)
        allBs = positive_is_a[i:(i+group_size), 1:].repeat(1, d)

        embA = torch.gather(embeddings, 0, allAs)
        embB = torch.gather(embeddings, 0, allBs)
        # one false negative for every pair where
        # A=0 and B=1 in any position

        bad = (
                torch.all(embA == embB, dim=1).to(dtype=torch.int32) )  
        falseneg += torch.sum(
                torch.any((~embA) & embB, dim=1).to(dtype=torch.int32) |
                bad)

        # loss is tricky with unseens...
        loss_pos += torch.sum( (embA == 0) & (embB == 1) ).to(dtype=torch.int32)

    for i in range(0, negative_is_a.size(0), group_size):
        # negative samples
        # same algorithm, just with true negatives
        allAs = negative_is_a[i:(i+group_size), :1].repeat(1, d)
        allBs = negative_is_a[i:(i+group_size), 1:].repeat(1, d)
        embA = torch.gather(embeddings, 0, allAs)
        embB = torch.gather(embeddings, 0, allBs)
        falsepos += torch.sum(looks_like_is_a(embA, embB).to(dtype=torch.int32) )  


    # now parse
    truepos = len(positive_is_a) - falseneg
    trueneg = len(negative_is_a) - falsepos

    # ...loss_neg is literally FP times beta
    return tuple(tensor.item() for tensor in (truepos, falseneg, falsepos, trueneg, loss_pos))

