#!python3
"""
node2binary.py

This is where things happen.
We need to:
    1. Load data
    2. Call run_model
    3. Evaluation

"""

import os
import sys
import random
import time
import torch
import click
import numpy as np



#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

from util.maybe_cuda import maybe_cuda
from util.run_model_f1_with_loss import run_model
from util.evaluate_embedding_with_loss import evaluate_embedding_with_loss, looks_like_is_a
from util.get_negative_samples import fast_accurate_negative_samples
from util.sgd_with_loss_and_ns import rng
from util.get_similar_pairs import get_similar_pairs
from util.graph_processing import save_embeddings, load_embeddings, load_labels, split_train_test_graph
#from util.evaluation import NodeClassification, LinkPrediction
from util.node_classification_subsample import NodeClassification
from util.link_prediction_subsample import LinkPrediction
from util.Leiden_hierarchical_forest import get_hypernym_edges
import matplotlib.pyplot as plt
from util.parse_file import parse_file, unpack, unpack_subset # parsing
random.seed(522)

@click.command(no_args_is_help=True)
@click.argument("edgelist_file", required=True, type=click.Path(exists=True))
@click.option("--is_weighted", is_flag=True, default=False)
@click.option("--labels_path",default=None)
@click.argument("dimension", required=True, type=int)
@click.argument("alpha", required=True, type=int)
@click.argument("beta", required=False, default=1, type=int)
@click.argument("pos_sim_weight", required=False, default=1, type=float)
@click.argument("neg_sim_weight", required=False, default=1, type=int)
@click.argument("learning_rate", required=True, type=float)
@click.argument("learning_bias", required=True, type=float)
@click.argument("neg_samp_multiplier", required=False, default=1, type=int)
@click.argument("neg_sim_multiplier", required=False, default=0, type=int)
@click.option("--add_edgelist_to_siblings", is_flag=True, default=False)
@click.option("--closed", default=True, is_flag=True, help="if true, assume dataset is already transitive closure")
@click.option("--random-init", default=False, is_flag=True, help="if true, initialize randomly instead of to all 0's")
@click.option("--stop-width", default=50, type=int)
@click.option("--iterations", default=500, type=int)
@click.option("--batchsize", default=0, type=int, help="batchsize must be greater than or equal to 500")
@click.option("--verbose", default=1, type=float)
@click.option("--subsample_ratio", required=False,  default=1, type=float)
@click.option("--gradient2", is_flag=True, default=False)
@click.option("--trees", default=1, type=int, help="if trees>1 then it is a forest")
@click.option("--depth", required=True, type=int)
@click.option("--q", required=False, type=float)
@click.option("--task", type=str, default=None)
@click.option("--testing_ratio", type=float)
@click.option('--binary_operator', type=click.Choice(["average", "hadamard", "l1", "l2"], case_sensitive=False), default= "hadamard")
@click.option("--seed", type=int, default=42) 

def main( edgelist_file, is_weighted, labels_path, dimension, alpha, beta, pos_sim_weight, neg_sim_weight, learning_rate, learning_bias, neg_samp_multiplier, neg_sim_multiplier, add_edgelist_to_siblings, closed, random_init, stop_width, iterations, batchsize, verbose, subsample_ratio, gradient2, trees, depth, q, task, testing_ratio, binary_operator, seed ):

    # print current time in green
    print("\u001b[32;1m")
    print("STARTING at time:", time.asctime())
    print("\u001b[m")
    start_time = time.time()

    """
    Step 1: Load Data
    """

    # extract dataset name from the data_path name
    datafile_abs_path = os.path.abspath(edgelist_file)
    base_name, ext = os.path.splitext(os.path.basename(datafile_abs_path))

    # Generate forest based on the number of trees and depth
    graph_forest = "data/Using_Leiden/" + base_name + "_t" + str(trees) + "_d" + str(depth) + ".forest"

    # if not os.path.exists(graph_forest):
    # Always generate to get exact tree building time
    get_hypernym_edges(edgelist_file, trees, depth, is_weighted = is_weighted)
    
    pairs = parse_file(graph_forest)
    #pairs = random.sample(pairs, int(0.1*len(pairs)))
    print("parsed", len(pairs), "hierarchy pairs")
    #print("pairs:", pairs)
    siblings_tree = "data/Using_Leiden/" + base_name + "_t" + str(trees) + "_d" + str(depth) + ".siblings"
    siblings_pairs = parse_file(siblings_tree)
    
    #subset_siblings_pairs = random.sample(siblings_pairs, int(0.05*len(siblings_pairs)))
    #print("leaf_siblings_pairs:", leaf_siblings_pairs)
    
    # use sets to clean out duplicates, but convert to list at the end
    (words, num_words, word_to_index, pair_numbers) = unpack(pairs)
    #print("words:", words)
    #print("word_to_index:", word_to_index)
    #print("Entities:", num_words)

    splittable_pairs = list((A,B) for (A,B) in pair_numbers if A != B) # copy it
    random.shuffle(splittable_pairs)
    num_dupes = 0

    pair_tensor = torch.tensor(splittable_pairs, dtype=torch.int64); # size: (number of pairs, 2)
    #print(pair_tensor)

    # Load sibling data
    sibling_pairs, sibling_pair_numbers = None, None
    #sibling_pairs = sorted(set(subset_siblings_pairs))
    sibling_pairs = sorted(set(siblings_pairs))
    try:
        sibling_pair_numbers = [(word_to_index[a1], word_to_index[a2]) for (a1, a2) in sibling_pairs]
        print("parsed", len(sibling_pair_numbers), "sibling pairs")   
    except KeyError as e:
        # something helpful, like...
        raise Exception(f"Siblings file contains {e.args[0]}, which isn't a word in the data file")
    
    if add_edgelist_to_siblings:
        edgelist_pairs = parse_file(edgelist_file)
        edgelist_pairs_filtered = []  
        try:
            if len(list(edgelist_pairs[0])) > 2:
                for (n1,n2,_) in edgelist_pairs:
                    if n1 > n2:
                        continue
                    edgelist_pairs_filtered.append((n1,n2)) 
    
            else:
                for (n1,n2) in edgelist_pairs:
                    if n1 > n2:
                        continue
                    edgelist_pairs_filtered.append((n1,n2)) 
            edgelist_pair_numbers = [(word_to_index[a1], word_to_index[a2]) for (a1, a2) in edgelist_pairs_filtered]
            print("parsed", len(edgelist_pair_numbers), "edgelist pairs")   
        except KeyError as e:
            # something helpful, like...
            raise Exception(f"Edgelist file contains {e.args[0]}, which isn't a word in the data file")
        sibling_pair_numbers += edgelist_pair_numbers
    
    #sibling_pair_numbers = edgelist_pair_numbers
    print("Final sibling pairs count:", len(sibling_pair_numbers))
    train_siblings_pos = torch.tensor(sibling_pair_numbers, dtype=torch.int64, device=maybe_cuda)

    # Create training and validation dataset
    #full_tensor = pair_tensor
    train_pairs = pair_tensor
    val_pairs = pair_tensor
    val_negatives = None # run_model will create them
    print("training on", len(train_pairs), "pairs")
    # default to size of train pairs
    if batchsize <= 0:
        batchsize = len(train_pairs)

    leaf_nodes = dict()
    for index in range(len(words)):
         try:
             if type(int(words[index])) in (int, np.int32, np.int64):
                 leaf_nodes[words[index]] = index
         except:
             pass
    print("number of leaf nodes: ", len(leaf_nodes.keys()))
    #print("\nLeaf nodes dictionary: ", leaf_nodes_mapped_original)    

    """
    Step 2: Call run_model
    """

    if task == "NodeClassification":
        # Load labels
        labels = load_labels(os.path.abspath(labels_path))
    else:
        labels = None
    
    final_embeddings = run_model(edgelist_file, is_weighted, words, dimension, labels, train_pairs.to(device=maybe_cuda), train_siblings_pos.to(device=maybe_cuda), val_pairs.to(device=maybe_cuda),
            alpha=alpha, beta=beta, pos_sim_weight=pos_sim_weight, neg_sim_weight=neg_sim_weight, 
            learning_rate=learning_rate, learning_bias=learning_bias,
            neg_samp_multiplier=neg_samp_multiplier, neg_sim_multiplier=neg_sim_multiplier, graph_node_to_index=leaf_nodes,
            start_from_zero = not random_init,
            stop_width = stop_width, max_iterations = iterations, batchsize = batchsize,
            val_negatives = val_negatives, task = task,
            verbose = verbose, subsample_ratio=subsample_ratio, gradient2 = gradient2, testing_ratio=testing_ratio, binary_operator=binary_operator, seed=seed)

    #Output_filename
    input_file_abs_path = os.path.abspath(edgelist_file)
    base_name, ext = os.path.splitext(os.path.basename(input_file_abs_path))
    output_filename = "data/Using_Leiden/" + base_name + "_t" + str(trees) + "_d" + str(depth) + ".embeddings.txt"

    # save embeddings in a text file
    save_embeddings(final_embeddings, output_filename, leaf_nodes)

    '''
    """
    Step 3: Evaluation
    """
    if task == "NodeClassification":
        # load embeddings and labels
        embedding_look_up = load_embeddings(os.path.abspath(output_filename))
        labels = load_labels(os.path.abspath(labels_path))

        node_list = list(embedding_look_up.keys())
        acc, micro, macro = NodeClassification(embedding_look_up, node_list, labels, testing_ratio, seed)
        print('\n\n')
        print('#' * 9 + ' NODE-CLASSIFICATION ' + '#' * 9)
        print(f"Accuracy: {acc}, Micro F1: {micro}, Macro F1: {macro}")
        print('#' * 50)
    
    if task == "LinkPrediction":
        # get data split
        input_file_to_use = os.path.abspath(edgelist_file)
        G, G_train, testing_pos_edges, _ = split_train_test_graph(input_file_to_use,
                                                    seed, testing_ratio, is_weighted)
        # load embeddings
        embedding_look_up = load_embeddings(os.path.abspath(output_filename))

        # test link prediction
        auc_roc, auc_pr, accuracy, f1 = LinkPrediction(embedding_look_up, G, G_train, 
                                                       testing_pos_edges, seed, binary_operator)
        print('#' * 9 + ' LINK PREDICTION ' + '#' * 9)
        print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
        print('#' * 50)


    print("\u001b[m") # reset colors
    '''

    total_time = time.time() - start_time
    
    print("\n\nOverall Time: {:02d}h {:02d}m {:02d}s".format(int(total_time / 3600), int(total_time / 60) % 60, int(total_time) % 60))


main()
