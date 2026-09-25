"""
run_model.py

Puts it all together!
"""

import time
import math
import os
from typing import Dict

import torch
import numpy as np
from .sgd_with_loss_and_ns import rng, initialize, get_gradient_and_loss, multithread_gradient
from .graph_processing import split_train_test_graph, load_embeddings
from .flip_probability import flip_prob
from .evaluate_embedding_with_loss import evaluate_embedding_with_loss
from .get_negative_samples import get_negative_samples, fast_negative_samples, fast_accurate_negative_samples
from .maybe_cuda import maybe_cuda
from .node_classification_subsample import NodeClassification
from .link_prediction_subsample import LinkPrediction

import matplotlib.pyplot as plt


def seconds():
    return time.time()

# plot binder loss
def plot_binder_loss(binder_losses):
    plt.plot(binder_losses, label="Train")
    plt.legend()
    plt.title('Binder Train Loss plot')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plot_filename = os.path.join("loss_plots", "binder_train_loss_plot.png")
    plt.savefig(plot_filename)
    #plt.show()

# plot similarity loss
def plot_similarity_loss(sim_losses):
    plt.plot(sim_losses, label="Train")
    plt.legend()
    plt.title('Similarity Train Loss plot')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plot_filename = os.path.join("loss_plots", "similarity_train_loss_plot.png")
    plt.savefig(plot_filename)
    #plt.show()

# plot total loss
def plot_total_loss(total_losses):
    plt.plot(total_losses, label="Train")
    plt.legend()
    plt.title('Total Train Loss plot')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    lot_filename = os.path.join("loss_plots", "total_train_loss_plot.png")
    lt.savefig(plot_filename)
    #plt.show()


def run_model(edgelist_file, is_weighted, words, dimension, labels, train_pairs, train_siblings_pos, val_pairs, *,
        alpha, beta, pos_sim_weight, neg_sim_weight, learning_rate, learning_bias, neg_samp_multiplier, neg_sim_multiplier, graph_node_to_index: Dict,
        start_from_zero = True,
        stop_width = 50, max_iterations = 100, batchsize,
        val_negatives = None, task,
        verbose = 1, subsample_ratio, gradient2 = False, testing_ratio,  binary_operator, seed = 0):
    """
    Step 1: Initialize the model

    "A is-a B" means "B is a hypernym of A" so B is in hypernyms_of[A]
    """
    global_start_time_limit = seconds()

    # if we're given no validation pairs, it's just a reconstruction task
    reconstruction = (len(val_pairs) == 0)

    # Create the embeddings.
    embeddings = initialize(len(words), dimension, start_from_zero)

    train_pairs_set = frozenset((int(p[0]), int(p[1])) for p in train_pairs.to(device="cpu")) # for negative sampling, to check if a pair is positive in O(1) time
    val_pairs_set = frozenset((int(p[0]), int(p[1])) for p in val_pairs.to(device="cpu"))
    train_val_pairs_set = val_pairs_set.union(train_pairs_set)

    unknown_set = set(range(len(words))) - (
            set(p[0] for p in train_pairs_set).union(set(p[1] for p in train_pairs_set)) 
    )
    print("There are {} unknown entities".format(len(unknown_set)))
    for un in unknown_set:
        embeddings[un,:] = -1 # an "undefined" value

    # Negative samples for validation set
    # we can do the slow way
    # I also realized I can fool get_negative_samples into corrupting val_pairs but checking all pairs
    print("generating negative validation of size", len(train_pairs if reconstruction else val_pairs))
    if val_negatives is None:
        val_negatives = fast_accurate_negative_samples(len(words), train_pairs if reconstruction else val_pairs, train_val_pairs_set, 1)
    print("done")

    train_siblings_neg = fast_accurate_negative_samples(len(words), train_siblings_pos, train_val_pairs_set, neg_sim_multiplier)


    # batch size
    if batchsize <= 0 or batchsize > len(train_pairs):
        batchsize = len(train_pairs)

    f1_scores = []
    accuracies = []
    val_losses = []
    binder_losses = []
    sim_losses = []
    total_losses = []
    best_embedding = None
    best_loss = 1e300
    best_f1_score = 0
    #best_accuracy = 0
    best_iteration = -1
    best_tuple = ()
    best_macro = 0
    best_auc_roc = 0

    '''
    if task == "LinkPrediction":
       # get data split
       input_file_to_use = os.path.abspath(edgelist_file)
       G, G_train, testing_pos_edges, _ = split_train_test_graph(input_file_to_use,
                                            seed, testing_ratio, is_weighted)
    '''

    """
    Step 2: Run the algorithm
    """

    finished = False
    iteration = 1
    last_print_time = None
    check_running_time = True
    start_time = seconds()
    try:
        while not finished:
            # so if verbose is 0.1 it prints only every 10 times
            # 1e-12 avoids double printing due to float errors
            verbose_level1 = ((iteration * verbose + 1e-12) % 1) < verbose
            # Print the header information.
            if verbose_level1:
                print("----- Iteration {} -----".format(iteration))
                if last_print_time is not None:
                    print("time since last update: {:.3g}s".format(seconds() - last_print_time))
                last_print_time = seconds()

            sub_iter_len = math.ceil(len(train_pairs)/batchsize)
            train_loss = 0
            train_sibling_loss = 0
            for i in range(sub_iter_len):
                if i == (sub_iter_len-1):
                    train_minibatch = train_pairs[i*batchsize : len(train_pairs)]
                else:
                    train_minibatch = train_pairs[i*batchsize : (i+1)*batchsize]
                t = seconds()

                # Get negative samples
                # We include both the random samples and flipped versions of positive pairs
                # (if A is-a B then put B is-not-an A in negatives)
                #negative_samples = get_negative_samples(words, pairs, pairs_set, neg_samp_multiplier)
                negative_samples = fast_negative_samples(len(words), train_minibatch, neg_samp_multiplier, device = maybe_cuda)
                train_pairs_flip = torch.empty(train_minibatch.size(), dtype=torch.int64, device=negative_samples.device)
                train_pairs_flip[:, 0] = train_minibatch[:,1]
                train_pairs_flip[:, 1] = train_minibatch[:,0]
                negative_samples = torch.cat((negative_samples, train_pairs_flip))
                neg_sample_time = seconds() - t
                # print time
                if verbose_level1:
                    print("Negative samples: {:.3g}s".format(seconds() - t))

                # We have our samples now
                # Run the SGD algorithm
                t = seconds()
                (gradient, partial_loss, sib_loss, train_siblings_loss) = get_gradient_and_loss(embeddings, dimension, train_minibatch, negative_samples, train_siblings_pos, train_siblings_neg,
                        alpha = alpha, beta = beta, pos_sim_weight=pos_sim_weight, neg_sim_weight=neg_sim_weight, gradient2 = gradient2,
                        use_tensors=True)

                train_loss += partial_loss
                train_sibling_loss += sib_loss
                gradient_time = seconds() - t
                if verbose >= 2:
                    for word_index in range(len(words)):
                        print("{:15s}".format(words[word_index]), " embedded as ", embeddings[word_index], " gradient is ", gradient[word_index])

                # Flip the words based on this gradient
                # I don't expect this to be very slow
                t = seconds()
                #gradient_quarter = torch.max(gradient)
                probs = flip_prob(gradient.float() * learning_rate + learning_bias)
                # sample random floats
                flip = torch.rand(probs.size(), device=maybe_cuda) < probs
                #if verbose >= 2:
                    # The " ".join() prints strings separated by spaces.
                    # Here, it makes an F where we flip and _ where we don't.
                '''
                    for word_index in range(len(words)):
                        print("{:15s}".format(words[word_index]),
                            "flips", " ".join("F" if flip[i] == 1 else "_" for i in range(len(flip))),
                            "Flip Probabilities: ", np.around(probs, 2) )
                    '''
                    #pass
                flip[embeddings < 0] = 0 # if embeddings are missing (unseen entity) don't even try to flip them
                # now use the exclusive or function to actually flip them
                # not sure why the trailing underscore means "in place"
                embeddings.bitwise_xor_(flip)
                flip_time = seconds() - t
                if verbose_level1:
                    print("Times: NegativeSamples = {:.3g}s, Gradient = {:.3g}s, Flip = {:.3g}s".format(neg_sample_time, gradient_time, flip_time))

            # Evaluate the model
            # if we're doing reconstruction, use train pairs
            # if we're doing prediction, use validation pairs
            t = seconds()
            (TP, FN, FP, TN, loss_pos) = evaluate_embedding_with_loss(embeddings,
                                              train_pairs if reconstruction else val_pairs,
                                              val_negatives)
            val_loss = loss_pos * alpha + FP * beta # our negative loss *literally* is FP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0 # ?!?
            recall = TP / (TP + FN)
            neg_recall = TN / (TN + FP)
            balanced_precision = recall / (recall + 1.0 - neg_recall) if (recall + 1.0 - neg_recall) > 1e-8 else 0
            #f1_score = 2.0 / (1.0/balanced_precision + 1.0/recall) if recall > 0 else 0
            f1_score = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            accuracy = (recall + neg_recall) / 2.0
            if verbose_level1:
                train_template = "TRAINING:   TP = {}   FN = {}  VALIDATION:  FP = {}  (FP out of {}). Loss = {}"
                val_template = "VALIDATION: TP = {}   FN = {}   FP = {}  (FP out of {}). Loss = {}"
                print((train_template if reconstruction else val_template).format(TP, FN, FP, len(val_negatives), val_loss))
                print("Sibling loss:", int(train_sibling_loss))
                print("Sibling similarity: {:.2f}%".format(
                    # train_sibling_loss is a raw gamma * number of bits that differ
                    # / gamma means it's the actual number of different bits
                    # / len(train_siblings) averages over sibling edges
                    # / dimension changes it from "number of bits" to "fraction of bits"
                    # 100 - that*100 makes it a similarity percentage
                    100 - train_siblings_loss / (dimension * len(train_siblings_pos) * max(pos_sim_weight, 1)) * 100
                ))
                print("Density of 1's: {:.2f}%".format(100 * torch.sum(embeddings).item() / torch.numel(embeddings)))
                print("F1 score = {:.3f} • Accuracy = {:.3f}    (took {:.3g}s)".format(f1_score, accuracy, seconds() - t))
            accuracies.append(accuracy)
            f1_scores.append(f1_score)
            binder_losses.append(train_loss)
            sim_losses.append(train_sibling_loss)
            total_losses.append(train_loss + train_sibling_loss) 
            val_losses.append(val_loss)

            '''
            if accuracy > best_accuracy:
               best_embedding = embeddings.clone().detach()
               best_iteration = iteration
               best_accuracy = accuracy
               best_tuple = (TP, FN, FP, TN)
	    '''
            
            if iteration % (1/verbose) == 0:

                if check_running_time == True:
                    print ("\nTime taken before first evaulation: ", seconds()-start_time)
                    evaluation_start_time = seconds()

                ## Evaluate Task Performance

                # Filter out graph node embeddings
                vectors = embeddings.cpu().detach().numpy()
                embedding_look_up = {}
                for k, v in graph_node_to_index.items():
                    embedding_look_up[k] = vectors[v]

                if task == "NodeClassification":

                    node_list = list(embedding_look_up.keys())
                    # Load labels
                    acc, micro, macro = NodeClassification(embedding_look_up, node_list, labels, testing_ratio, subsample_ratio)
                    print("\n")
                    print('#' * 9 + ' NODE-CLASSIFICATION ' + '#' * 9)
                    print(f"Accuracy: {acc}, Micro F1: {micro}, Macro F1: {macro}")
                    print('#' * 50)
                    print("\n")

                    if macro > best_macro: # don't determine "best macro F1 score" super early
                 
                        best_embedding = embeddings.clone().cpu().detach()
                        best_macro = macro
                        best_iteration = iteration
                        print("best iteration: ", best_iteration)
                        print("best macro score: ", best_macro)
                        best_loss = val_loss
                        best_f1_score = f1_score
                        best_tuple = (TP, FN, FP, TN)

                if task == "LinkPrediction":

                    Keys = list(embedding_look_up.keys())
                    Keys_int = []
                    for i in range(len(Keys)):
                        Keys_int.append(Keys[i])
                        
                    sorted_Keys = sorted(Keys_int)
                    embs = []
                    for i in sorted_Keys:
                        embs.append(embedding_look_up[i])
                    
                    input_folder = "link_prediction"
                    # test link prediction
                    auc_roc = LinkPrediction(edgelist_file, input_folder, is_weighted, embs, binary_operator, testing_ratio, subsampling_ratio = subsample_ratio)
                    print('#' * 9 + ' LINK PREDICTION ' + '#' * 9)
                    print("Binary Operator: ", binary_operator, "AUC-ROC: ", auc_roc)
                    #for i in range(len(auc_roc_results)):
                    #    print("AUC-ROC: ", auc_roc_results[i])
                    #print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
                    print('#' * 50)
                    print("\n")

                    if auc_roc > best_auc_roc:        #auc_roc_results[1][1] > best_auc_roc: 
                 
                        best_embedding = embeddings.clone().cpu().detach()
                        best_auc_roc =  auc_roc      #auc_roc_results[1][1]
                        best_iteration = iteration
                        print("best iteration: ", best_iteration)
                        print("best auc_roc score: ", best_auc_roc)
                        best_loss = val_loss
                        best_f1_score = f1_score
                        best_tuple = (TP, FN, FP, TN)
                
                if check_running_time == True:
                    print ("\nTime taken for first evaulation: ", seconds()-evaluation_start_time)
                    check_running_time = False
            '''
            if val_loss < best_loss and iteration >= min(10, max_iterations - 1): # don't determine "best loss" super early
                best_embedding = embeddings.clone().detach()
                best_iteration = iteration
                best_loss = val_loss
                best_f1_score = f1_score
                best_tuple = (TP, FN, FP, TN)
            '''

            # determine if we are done...
            iteration += 1
            if iteration > max_iterations:
             	finished = True

            if (seconds() - global_start_time_limit) > 1800:
                print("Time limit of 30 minutes reached. Stopping early.", flush=True)
                finished = True
            
            if len(val_losses) >= 2*stop_width and sum(val_losses[-2*stop_width : -stop_width]) <= sum(val_losses[-stop_width:]):
                finished = True
            '''
            if len(accuracies) >= 2*stop_width and sum(accuracies[-2*stop_width : -stop_width]) >= sum(accuracies[-stop_width:]):
              	finished = True
            '''

        #plot_binder_loss(binder_losses)  
        #plot_similarity_loss(sim_losses)
        #plot_total_loss(total_losses) 
  
    except KeyboardInterrupt:
        print("Interrupted!")

    print("\u001b[36;1m")

    '''
    if verbose > 0:
        print("Beginning: ", f1_scores[:500])
        print("History: ", f1_scores[::25])
        print("Last: ", f1_scores[-500:])
        print()
    
    
    print("Best Validation Loss appeared on iteration {} with Loss {} and F1 score {:.4f} {}".format(best_iteration, best_loss, best_f1_score, str(best_tuple)))
    #print("Best Validation accuracy appeared on iteration {} with accuracy {:.4f} {}".format(best_iteration, best_accuracy, str(best_tuple)))
    print("\u001b[m")


    print("===== TRAINING LOSSES below =====")
    print(train_losses)
    print("===== VALIDATION LOSSES below =====")
    print(val_losses)
    '''


    # loss_log = open("losses_d_{}.log".format(time.strftime("%Y-%m-%d_%H-%M-%S")), "w")
    #for loss in losses:
        #print(loss, file=loss_log)
        #print(loss)

    # Return the embeddings so we can use them.
    return best_embedding




