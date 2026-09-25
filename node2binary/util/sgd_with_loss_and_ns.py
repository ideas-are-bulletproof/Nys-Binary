# sgd.py
# 
# These functions compute our discrete SGD.

import numpy as np
from numpy.random import Generator, default_rng
import torch
rng = default_rng()

from .maybe_cuda import maybe_cuda

import threading

"""
Returns a dictionary mapping each word to...
if use_zero:
    ...an array of size dimension, with all zeros
else:
    ...an array of size dimension, with random bits
"""
def initialize(num_words, dimension, use_zero=True):
    if use_zero:
        return torch.zeros((num_words, dimension), dtype=torch.bool, device=maybe_cuda)
    else:
        return torch.randint(0, 2, (num_words, dimension), dtype=torch.bool, device=maybe_cuda)

"""

Structure:

    embeddings: torch.tensor(), first dimension is words, second is bits
    dimension: integer
    hypernyms: torch.tensor(), size (h, 2) where h is the number of positive pairs
    not_hypernyms: torch.tensor(), size (nh, 2), nh = number of negatives


I recommend calling this using named parameters, like
get_gradient(embeddings = ..., dimension = 42,
    hypernums = ..., not_hypernyms = ...,
    alpha = 4, beta = 2)

This way, if we change the function (e.g. add another parameter) it won't mess up our function calls.


"""

def get_gradient_and_loss(embeddings, dimension, hypernyms, not_hypernyms, similar_pairs_pos, similar_pairs_neg,
        alpha = 1, beta = 1, pos_sim_weight = 1, neg_sim_weight = 1, gradient_bias = 100, gradient2 = False, use_tensors=True):
    """
    note:
    This code uses heavy use of multiplication as a way to do AND operators.
    For instance, if I want A=0 and B=1, then I do B * (1 - A). Why?

     A | B | 1 - A | B * (1 - A)
    ---+---+-------+------------
     0 | 0 |   1   |      0
     0 | 1 |   1   |      1     <<< this is what we want!
     1 | 0 |   0   |      0
     1 | 1 |   0   |      0


    """

    # Set the embeddings to all zero.

    #print("embeddings:", embeddings[0:5])
    #print("hypernyms:", hypernyms[0:5])
    #print("not_hypernyms:", not_hypernyms[0:5])
    #print("similar_pairs:", similar_pairs[0:5])
    
    torch.cuda.empty_cache() 

    if use_tensors:
        d = embeddings.size(1)
        # these are not that big
        gradient_pos = torch.zeros(embeddings.size(0), d, dtype=torch.int32, device=maybe_cuda)
        gradient_neg = torch.zeros(embeddings.size(0), d, dtype=torch.int32, device=maybe_cuda)
        gradient_sim_pos = torch.zeros(embeddings.size(0), d, dtype=torch.int32, device=maybe_cuda)

        # Process this in pseudo-minibatches to avoid excessive GPU memory usage
        # (this is NOT the minibatch parameter)
        group_size = 100
        # reason:
        # 10000 pairs * 500 (maximum reasonable dimension) * 4 bytes per pos
        # = 20'000'000 bytes per array.
        # actually there are multiple copies but it's still well under 1GB

        loss = 0
        sibling_loss = 0
        pos_sib_loss = 0

        for i in range(0, hypernyms.size(0), group_size):
            allAs = torch.tile(hypernyms[i:(i+group_size), :1], (1, d)) 
            allBs = torch.tile(hypernyms[i:(i+group_size), 1:], (1, d)) 

            embA = torch.gather(embeddings, 0, allAs).to(dtype=torch.int32)
            embB = torch.gather(embeddings, 0, allBs).to(dtype=torch.int32)
            gradient_pos.scatter_add_(0, allAs, embB * (1 - 2 * embA) )
            gradient_pos.scatter_add_(0, allBs, (1 - embA) * (2*embB - 1) )

            loss += alpha * torch.sum(embB * (1 - embA)).item()

            # recycling
            embA.resize_(0)
            embB.resize_(0)
            allAs.resize_(0)
            allBs.resize_(0)

        for i in range(0, not_hypernyms.size(0), group_size):
            # negative samples
            notAs = torch.tile(not_hypernyms[i:(i+group_size), :1], (1, d)) 
            notBs = torch.tile(not_hypernyms[i:(i+group_size), 1:], (1, d)) 
            embA = torch.gather(embeddings, 0, notAs).to(dtype=torch.int32)
            embB = torch.gather(embeddings, 0, notBs).to(dtype=torch.int32)
            a_0_b_1 = embB * (1 - embA)
            good_pairs = torch.sum(a_0_b_1, dim=1, keepdim=True)

            is_fp = torch.tile(good_pairs == 0, (1, d))
            is_close = torch.tile(good_pairs == 1, (1, d))

            # false positives (actually may not be false...)
            gradient_neg.scatter_add_(0, notAs, embA * embB * is_fp)
            gradient_neg.scatter_add_(0, notBs, (1-embA) * (1-embB) * is_fp)
            # one away from false positives
            gradient_neg.scatter_add_(0, notAs, -a_0_b_1 * is_close)
            gradient_neg.scatter_add_(0, notBs, -a_0_b_1 * is_close)

            loss += beta * torch.sum(
                   # torch.any(B * (1-A)) returns rows where (A,B) is a True Neg
                   # so we do 1 - that_vector to get negative loss
                   ~torch.any(embB & (1 - embA), dim=1)
                   ).item()

            # clear out the tmp arrays
            for tmp in (embA, embB, notAs, notBs, a_0_b_1, good_pairs, is_close, is_fp):
                tmp.resize_(0)
        
        for i in range(0, similar_pairs_pos.size(0), group_size):
            # similar pairs
            simAs = torch.tile(similar_pairs_pos[i:(i+group_size), :1], (1,d))
            simBs = torch.tile(similar_pairs_pos[i:(i+group_size), 1:], (1,d))
            # get the embeddings
            embA = torch.gather(embeddings, 0, simAs)
            embB = torch.gather(embeddings, 0, simBs)
            
            # this quantity |emb(a) - emb(b)| appears a lot
            difference = torch.logical_xor(embA, embB)
            # loss is just the number of bits that are different
            sibling_loss += pos_sim_weight * torch.sum(difference)
            pos_sib_loss += pos_sim_weight * torch.sum(difference)

            # the gradient of hamming distance is 2*|a-b| - 1
            # (returns -1, don't flip, if they're equal
            # and +1, please flip, if they're opposite)
            difference = 2 * difference.to(dtype=torch.int32) - 1

            gradient_sim_pos.scatter_add_(0, simAs, difference)
            gradient_sim_pos.scatter_add_(0, simBs, difference)

            # reset the tensors
            for recycle in (simAs, simBs, embA, embB, difference):
                recycle.resize_(0)
        
        
        gradient_sim = gradient_sim_pos * pos_sim_weight

        gradient_pos *= alpha
        gradient_neg *= beta
        
        return (gradient_pos + gradient_neg + gradient_sim, loss, sibling_loss, pos_sib_loss)
    


# Puts the gradient in the given array
def put_gradient(array, index, *args, **kwargs):
    array[index], _ = get_gradient(*args, **kwargs)

def multithread_gradient(num_threads, embeddings, dimension, hypernyms, not_hypernyms, alpha = 1, beta = 1):
    threads = [None for _ in range(num_threads)]
    results = [None for _ in range(num_threads)]
    for i in range(num_threads):
        threads[i] = threading.Thread(target=put_gradient, args=(
            results, i,
            embeddings, dimension,
            hypernyms[i::num_threads],
            not_hypernyms[i::num_threads]),
                                      kwargs={ "alpha": alpha, "beta": beta })
        threads[i].start()

    for i in range(num_threads):
        threads[i].join()

    # sum all the threads
    return {
            word: sum(result[word] for result in results)
            for word in embeddings
    }

def failed_gradient(embeddings, dimension, hypernyms_of, not_hypernyms_of, *,
        words_to_train = None,
        alpha = 1, beta = 1, learning_rate = 1,
        debug = False):
    # if no words_to_train specified, use all of them
    if words_to_train is None:
        words_to_train = list(embedding_dict.keys())

    for word in words_to_train:
        if debug:
            print("\nWord:", word)
        delta_pos = np.zeros(dimension, np.int32)
        delta_neg = np.zeros(dimension, np.int32)

        # positive samples
        # we'll multiply by alpha at the end
        plusOrMinus1 = (1 - 2*embeddings[word]) # this is the same in every iteration
        for hyper in hypernyms_of[word]:
            # hyper is "B", word is "A"
            # this computes B * (1 - 2*A)
            delta_pos += embeddings[hyper] * plusOrMinus1

        if debug:
            print("positive term:", delta_pos)

        """
        negative samples, part 1:
        I'm trying to be "efficient", by just counting the number of times a word is 
        """
        num_equal = sum(1
                for not_hyper in not_hypernyms_of[word]
                if np.all(np.bitwise_and(embeddings[not_hyper], embeddings[word]) == embeddings[word])
                )
        delta_neg += np.ones(dimension, np.int32) * num_equal

        if debug:
            print("first negative term:", delta_neg)

        # negative samples, part 2
        # explanation:
        # Hamming distance is the number of bits that are different.
        # XOR returns a 1 in every position where the bits are different.
        # Thus, summing the XOR gives the hamming distance.
        for not_hyper in not_hypernyms_of[word]:
            # we're using this in two places...
            the_xor = np.bitwise_xor(embeddings[word], embeddings[not_hyper])
            # once to compute the Hamming distance...
            if np.sum(the_xor) == 1:
                # and once to update the gradient
                delta_neg -= the_xor

        if debug:
            print("final negative term:", delta_neg)

        # now we're done.
        gradient[word] = alpha * delta_pos + beta * delta_neg
        if debug:
            print("gradient[{}] =".format(word), gradient[word])
    return gradient


