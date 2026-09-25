# import packages

import networkx as nx
import numpy as np
import sys
import os
import copy
from collections import OrderedDict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)



def read_graph(input_file, is_weighted):
    '''
    Reads the input network in networkx.
    '''

    if is_weighted:
        G = nx.read_edgelist(input_file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
            
    G = G.to_undirected()
    
    return G


def read_embedding_file(file_path):

    embeddings = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            parts = line.strip().split()
            node_id = parts[0]  # Node ID as string
            vector = [float(x) for x in parts[1:]]  # Convert the rest to float
            embeddings[node_id] = vector
            
    Keys = list(embeddings.keys())
    Keys_int = []
    for i in range(len(Keys)):
        Keys_int.append(int(Keys[i]))
        
    #print("Embedding dictionary keys:", Keys)
    sorted_Keys = sorted(Keys_int)
    #print("sorted_Keys:", sorted_Keys)
    embedding_matrix = []
    for i in sorted_Keys:
        embedding_matrix.append(embeddings[str(i)])
    
    return np.array(embedding_matrix, dtype = 'f'), sorted_Keys


def split_into_training_test_sets(g, test_set_ratio, subsampling_ratio):

    print("--> The number of nodes: {}, the number of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

    print("+ Getting the gcc of the original graph.")
    # Keep the original graph
    train_g = g.copy()
    train_g.remove_edges_from(nx.selfloop_edges(train_g)) # remove self loops
    train_g = train_g.subgraph(max(nx.connected_components(train_g), key=len))
    if nx.is_frozen(train_g):
        train_g = nx.Graph(train_g)
    print("\t- Completed!")

    num_of_nodes = train_g.number_of_nodes()
    nodelist = list(train_g.nodes())
    edges = list(train_g.edges())
    num_of_edges = train_g.number_of_edges()
    print("--> The number of nodes: {}, the number of edges: {}".format(num_of_nodes, num_of_edges))
    
    if subsampling_ratio != 0:
        print("+ Subsampling initialization.")
        subsample_size = int(subsampling_ratio * num_of_nodes)
        while( subsample_size < train_g.number_of_nodes() ):
            remove_size = (
                # get closer to the goal
                (train_g.number_of_nodes() - subsample_size) // 2  +
                # also make progress when close to goal
                1 + subsample_size // 100
            )
            chosen = np.random.choice(list(train_g.nodes()), size=remove_size)
            train_g.remove_nodes_from(chosen)
            train_g = train_g.subgraph( max(nx.connected_components(train_g), key=len) )

            if nx.is_frozen(train_g):
                train_g = nx.Graph(train_g)
            print(subsample_size, train_g.number_of_nodes())

    print("+ Relabeling.")
    node2newlabel = {node: str(nodeIdx) for nodeIdx, node in enumerate(train_g.nodes())}
    train_g = nx.relabel_nodes(G=train_g, mapping=node2newlabel, copy=True)
    print("\t- Completed!")

    nodelist = list(train_g.nodes())
    edges = list(train_g.edges())
    num_of_nodes = train_g.number_of_nodes()
    num_of_edges = train_g.number_of_edges()
    print("--> The of nodes: {}, the number of edges: {}".format(num_of_nodes, num_of_edges))
    
    print("+ Splitting into train and test sets.")
    test_size = int(test_set_ratio * num_of_edges)

    test_g = nx.Graph()
    test_g.add_nodes_from(nodelist)

    count = 0
    idx = 0
    perm = np.arange(num_of_edges)
    while(count < test_size and idx < num_of_edges):
        if count % 10000 == 0:
            print("{}/{}".format(count, test_size))
        # Remove the chosen edge
        chosen_edge = edges[perm[idx]]
        train_g.remove_edge(chosen_edge[0], chosen_edge[1])
        #if True:
        if chosen_edge[1] in nx.connected._plain_bfs(train_g, chosen_edge[0]):
            test_g.add_edge(chosen_edge[0], chosen_edge[1])
            count += 1
        else:
            train_g.add_edge(chosen_edge[0], chosen_edge[1])

        idx += 1
    if idx == num_of_edges:
        raise ValueError("There are no enough edges to sample {} number of edges".format(test_size))
    else:
        print("--> Completed!")

    if count != test_size:
        raise ValueError("Enough positive edge samples could not be found!")

    # Generate the negative samples
    print("\+ Generating negative samples")
    count = 0
    negative_samples_idx = [set() for _ in range(num_of_nodes)]
    negative_samples = []
    while count < 2*test_size:
        if count % 10000 == 0:
            print("{}/{}".format(count, 2*test_size))
        uIdx = np.random.randint(num_of_nodes-1)
        vIdx = np.random.randint(uIdx+1, num_of_nodes)

        if vIdx not in negative_samples_idx[uIdx]:
            negative_samples_idx[uIdx].add(vIdx)

            u = nodelist[uIdx]
            v = nodelist[vIdx]

            negative_samples.append((u,v))

            count += 1

    train_neg_samples = negative_samples[:test_size]
    test_neg_samples = negative_samples[test_size:test_size*2]

    return train_g, test_g, train_neg_samples, test_neg_samples


def extract_feature_vectors_from_embeddings(edges, embeddings, binary_operator):

    features = []
    for i in range(len(edges)):
        edge = edges[i]
        vec1 = embeddings[int(edge[0])]
        vec2 = embeddings[int(edge[1])]

        if binary_operator == "average":
            value = [(g + h) / 2.0 for g, h in zip(vec1, vec2)]
        
        elif binary_operator == "hadamard":
            value = np.multiply(vec1, vec2)

        elif binary_operator == "l1":
            value = np.abs(np.array(vec1)-np.array(vec2))

        elif binary_operator == "l2":
            value = (np.array(vec1)-np.array(vec2)) ** 2

        else:
            raise ValueError("Invalid operator!")

        features.append(value)

    features = np.asarray(features)

    return features

def split(graph_path, is_weighted, output_folder, test_set_ratio, subsampling_ratio):

    # Read the network
    print("Graph is being read!")
    g = read_graph(graph_path, is_weighted)

    train_g, test_g, train_neg_samples, test_neg_samples = split_into_training_test_sets(
        g, test_set_ratio, subsampling_ratio)

    print("Train ratio: {}, #: {}".format(train_g.number_of_edges()/float(g.number_of_edges()), train_g.number_of_edges()))
    print("Test ratio: {}, #: {}".format(test_g.number_of_edges()/float(g.number_of_edges()), test_g.number_of_edges()))

    save_path = output_folder + "/"+ os.path.splitext(os.path.basename(graph_path))[0]

    nx.write_gml(train_g, save_path + "_gcc_train.gml")
    nx.write_edgelist(train_g, save_path + "_gcc_train.edgelist", data=['weight'])
    nx.write_gml(test_g, save_path + "_gcc_test.gml")

    np.save(save_path + "_gcc_train_negative_samples.npy", train_neg_samples)
    np.save(save_path + "_gcc_test_negative_samples.npy", test_neg_samples)


def LinkPrediction(graph_path, input_folder, is_weighted, embeddings, binary_operator, test_set_ratio, subsampling_ratio):

    print("-----------------------------------------------")
    print("Metric type: {}".format(binary_operator))
    print("-----------------------------------------------")

    graph_name = os.path.splitext(os.path.basename(graph_path))[0]
    if not os.path.exists(os.path.join(input_folder, graph_name + "_gcc_train.gml")):
        split(graph_path, is_weighted, input_folder, test_set_ratio, subsampling_ratio)

    train_g = nx.read_gml(os.path.join(input_folder, graph_name +"_gcc_train.gml"))
    train_neg_samples = np.load(os.path.join(input_folder, graph_name +"_gcc_train_negative_samples.npy"))

    train_samples = [list(edge) for edge in train_g.edges()] + list(train_neg_samples)
    train_labels = [1 for _ in train_g.edges()] + [0 for _ in train_neg_samples]
    print("train size: {}".format(len(train_labels)))

    train_features = extract_feature_vectors_from_embeddings(edges=train_samples,
                                                            embeddings=embeddings,
                                                            binary_operator=binary_operator)

    test_g = nx.read_gml(os.path.join(input_folder, graph_name+"_gcc_test.gml"))
    test_neg_samples = np.load(os.path.join(input_folder, graph_name+"_gcc_test_negative_samples.npy"))

    test_samples = [list(edge) for edge in test_g.edges()] + list(test_neg_samples)
    test_labels = [1 for _ in test_g.edges()] + [0 for _ in test_neg_samples]
    print("test size: {}".format(len(test_labels)))

    test_features = extract_feature_vectors_from_embeddings(edges=test_samples,
                                                            embeddings=embeddings,
                                                            binary_operator=binary_operator)
    
    clf = LogisticRegression(random_state=0, solver='liblinear', max_iter= 1000)
    clf.fit(train_features, train_labels)

    test_preds = clf.predict_proba(test_features)[:, 1]
    test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)

    print("For binary operator {}  test AUC-ROC score is {}\n".format(binary_operator, test_roc))

    
    return test_roc
