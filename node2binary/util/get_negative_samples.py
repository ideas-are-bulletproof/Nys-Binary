"""
get_negative_samples.py

The first function is a utility function.
There are 3 sampling methods implemented here:
    1. get_negative_samples is deprecated
    2. fast_negative_samples is a speedy version that does NOT check for true negativity.
    3. fast_accurate_negative_samples is a modified version that calls 2 repeatedly and uses a set difference, to hopefully speed up accurate negative sampling.

All functions create the negative samples by starting with a list of positive pairs and corrupting one element from each pair.
"""

import numpy as np
import torch
from .maybe_cuda import maybe_cuda
from .sgd_with_loss_and_ns import rng
#torch.manual_seed(522)

# word_position: 0 means word is the hyponym, 1 means it's the hypernym
def add_random_pair(num_words, negatives, pairs_set, word, word_position, root_object):
    # the root object has no negative samples
    if word == root_object:
        return

    success = False
    tries = 0
    while not success:
        other = rng.integers(num_words)
        pair = (word, other) if word_position == 0 else (other, word)
        # 3 ways to fail: (1) words are equal; (2) pair is positive; (3) pair is already used
        if (other != word) and (pair not in pairs_set) and (pair not in negatives):
            negatives.add(pair)
            success = True
        tries += 1
        # otherwise, try again
        if tries > 500:
            raise Exception("tried 500 times to corrupt {} in position {}".format(word, word_position))

    if tries > 5:
        #print("tried {} times to find a negative sample".format(tries))
        pass

# please don't use this one, it's outdated.
def get_negative_samples(words, pairs, pairs_set, root_object, neg_samp_multiplier):
    raise NotImplementedError("Deprecated")
    """
    negatives = set()
    # for odd multipliers
    if neg_samp_multiplier % 2 == 1:
        # corrupt one half
        for pair in pairs:
            if rng.random() < 0.5: # flip a coin
                # corrupt the hypernym
                add_random_pair(words, negatives, pairs_set, pair[0], 0, root_object)
            else:
                # corrupt the hyponym
                add_random_pair(words, negatives, pairs_set, pair[1], 1, root_object)

    # for each multiple of two
    for _ in range(0, neg_samp_multiplier, 2):
        # corrupt both halves
        for pair in pairs:
            # corrupt the hypernym
            add_random_pair(words, negatives, pairs_set, pair[0], 0, root_object)
            # corrupt the hyponym
            add_random_pair(words, negatives, pairs_set, pair[1], 1, root_object)

    # filter out anything that's not truly negative
    return negatives
    """

# this algorithm is super fast
def fast_negative_samples(num_words, pairs, neg_samp_multiplier, device):
    if neg_samp_multiplier % 2 != 0:
        raise ValueError("fast_negative_samples requires an even number of samples")

    half_multiplier = neg_samp_multiplier // 2
    num_pairs = len(pairs)
    half_total = num_pairs * half_multiplier

    # stretch out the tensor...
    output = pairs.repeat(neg_samp_multiplier, 1)
    # make the list of corrupted indices
    top_corrupted = torch.randint(0, num_words, (half_total,), dtype=torch.int32, device=device)
    bottom_corrupted = torch.randint(0, num_words, (half_total,), dtype=torch.int32, device=device)

    output[:half_total, 0] = top_corrupted
    output[half_total:, 1] = bottom_corrupted
    return torch.unique(output, dim=0).to(device=device)

# this should be fast but makes sure it is correct
def fast_accurate_negative_samples(num_words, pairs, pairs_set, neg_samp_multiplier):
    pairs_cpu = pairs.to(device='cpu')
    num_wanted = neg_samp_multiplier * len(pairs_cpu)
    negatives = set()

    # this probably could be improved
    while len(negatives) < num_wanted:
        # sigh...
        # trying to take advantage of array computation
        # this comes at the expense of understandability

        # first we generate some negative samples
        # then we find the unique elements
        batch = fast_negative_samples(num_words, pairs_cpu, 2, 'cpu')
        negatives.update( (int(n[0]), int(n[1])) for n in batch )

        # make sure they are actually negative.
        negatives -= pairs_set
        negatives = set(n for n in negatives if n[0] != n[1])

    # now randomly sample it
    # shuffle it and return the first len(pairs) items
    negatives = torch.tensor([n for n in negatives], dtype=torch.int64)
    #print("negatives has size", negatives.size())
    perm = torch.randperm(len(negatives), device=negatives.device)[:num_wanted]
    return negatives[perm,:].to(device=pairs.device)


