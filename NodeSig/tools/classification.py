import networkx as nx
import numpy as np
from collections import OrderedDict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy.spatial.distance import cdist
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from common import *

_score_types = ['micro', 'macro']

########################################################################################################################
parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--graph_path', type=str, required=True, help='Path of the graph in gml format'
)
parser.add_argument(
    '--emb_file', type=str, required=True, help='Path of embedding file'
)
parser.add_argument(
    '--emb_file_type', type=str, required=True, choices=["binary", "text"], help='File type'
)
parser.add_argument(
    '--output_file', type=str, required=True, help='Output result file'
)
parser.add_argument(
    '--shuffles_num', type=int, default=10, required=False, help='Number of shuffles'
)
parser.add_argument(
    '--train_ratio', type=str, required=False, default="custom",
    choices=["large", "all", "custom", "0.1", "0.5", "0.9"], help='Training ratios'
)
parser.add_argument(
    '--method', type=str, required=True,
    choices=["svm-chisquare", "svm-hamming", "svm-cosine"], help='Classification method'
)
########################################################################################################################


def detect_number_of_communities(nxg):
    # It is assumed that the labels of communities starts from 0 to K-1
    max_community_label = -1
    communities = nx.get_node_attributes(nxg, "community")
    # for node in nxg.nodes():
    for node in communities:
        comm_list = communities[node]
        if type(comm_list) is int:
            comm_list = [comm_list]

        c_max = max(comm_list)
        if c_max > max_community_label:
            max_community_label = c_max
    return max_community_label + 1


def get_node2community(nxg):

    node2community = nx.get_node_attributes(nxg, name='community')

    # for node in nxg.nodes():
    for node in node2community:
        comm = node2community[node]
        if type(comm) == int:
            node2community[node] = [comm]

    return node2community


def evaluate(graph_path, embedding_file, number_of_shuffles, training_ratios, classification_method, file_type="binary"):

    cache_size = 10240
    # Read the gml file
    g = nx.read_gml(graph_path)
    # A dictionary of node to community
    node2community = get_node2community(g)
    # Get the number of communities
    K = detect_number_of_communities(g)
    # Node list
    nodelist = [int(node) for node in node2community]
    # Number of nodes
    N = len(nodelist)
    # Read the embedding file
    x = read_embedding_file(file_path=embedding_file, nodelist=nodelist, file_type=file_type)

    label_matrix = [[1 if k in node2community[str(node)] else 0 for k in range(K)] for node in nodelist]
    label_matrix = csr_matrix(label_matrix)

    results = {}

    for score_t in _score_types:
        results[score_t] = OrderedDict()
        for ratio in training_ratios:
            results[score_t].update({ratio: []})

    print("+ Similarity matrix is begin computed!")
    if classification_method == "svm-hamming":
        sim = 1.0 - cdist(x, x, 'hamming')
    elif classification_method == "svm-cosine":
        sim = 1.0 - cdist(x, x, 'cosine')
        sim[np.isnan(sim)] = 0
        sim[np.isinf(sim)] = 0
    elif classification_method == "svm-chisquare":
        dist = cdist(x, x, 'hamming')
        dist[dist > 0.5] = 0.5
        sim = 1.0 - np.sqrt(2.0 - 2.0*np.cos(dist*np.pi))
    else:
        raise ValueError("Invalid classification method name: {}".format(classification_method))

    for train_ratio in training_ratios:

        for shuffleIdx in range(number_of_shuffles):

            print("Current train ratio: {} - shuffle: {}/{}".format(train_ratio, shuffleIdx+1, number_of_shuffles))

            # Shuffle the data
            shuffled_idx = np.random.permutation(N)
            shuffled_sim = sim[shuffled_idx, :]
            shuffled_sim = shuffled_sim[:, shuffled_idx]
            shuffled_labels = label_matrix[shuffled_idx]

            # Get the training size
            train_size = int(train_ratio * N)
            # Divide the data into the training and test sets
            train_sim = shuffled_sim[0:train_size, :]
            train_sim = train_sim[:, 0:train_size]
            train_labels = shuffled_labels[0:train_size]

            test_sim = shuffled_sim[train_size:, :]
            test_sim = test_sim[:, 0:train_size]
            test_labels = shuffled_labels[train_size:]

            # Train the classifier
            ovr = OneVsRestClassifier(SVC(kernel="precomputed", cache_size=cache_size, probability=True))

            ovr.fit(train_sim, train_labels)

            # Find the predictions, each node can have multiple labels
            test_prob = np.asarray(ovr.predict_proba(test_sim))
            y_pred = []
            for i in range(test_labels.shape[0]):
                k = test_labels[i].getnnz()  # The number of labels to be predicted
                pred = test_prob[i, :].argsort()[-k:]
                y_pred.append(pred)

            # Find the true labels
            y_true = [[] for _ in range(test_labels.shape[0])]
            co = test_labels.tocoo()
            for i, j in zip(co.row, co.col):
                y_true[i].append(j)

            mlb = MultiLabelBinarizer(range(K))
            for score_t in _score_types:
                score = f1_score(y_true=mlb.fit_transform(y_true),
                                 y_pred=mlb.fit_transform(y_pred),
                                 average=score_t)

                results[score_t][train_ratio].append(score)

    return results


def get_output_text(results, shuffle_std=False, detailed=False):

    num_of_shuffles = len(list(list(results.values())[0].values())[0])
    train_ratios = [r for r in list(results.values())[0]]
    percentage_title = " ".join("{0:.0f}%".format(100 * r) for r in list(results.values())[0])

    output = ""
    for score_type in _score_types:
        if detailed is True:
            for shuffle_num in range(1, num_of_shuffles + 1):
                output += "{} score, shuffle #{}\n".format(score_type, shuffle_num)
                output += percentage_title + "\n"
                for ratio in train_ratios:
                    output += "{0:.5f} ".format(results[score_type][ratio][shuffle_num - 1])
                output += "\n"

        output += "{} score, mean of {} shuffles\n".format(score_type, num_of_shuffles)
        output += percentage_title + "\n"
        for ratio in train_ratios:
            output += "{0:.5f} ".format(np.mean(results[score_type][ratio]))
        output += "\n"

        if shuffle_std is True:
            output += "{} score, std of {} shuffles\n".format(score_type, num_of_shuffles)
            output += percentage_title + "\n"
            for ratio in train_ratios:
                output += "{0:.5f} ".format(np.std(results[score_type][ratio]))
            output += "\n"

    return output


def print_results(results, shuffle_std, detailed=False):
    output = get_output_text(results=results, shuffle_std=shuffle_std, detailed=detailed)
    print(output)


def save_results(results, output_file, shuffle_std, detailed=False):

    with open(output_file, 'w') as f:
        output = get_output_text(results=results, shuffle_std=shuffle_std, detailed=detailed)
        f.write(output)


if __name__ == "__main__":

    # Parse the arguments
    args = parser.parse_args()

    graph_path = args.graph_path  # sys.argv[1]
    embedding_file = args.emb_file  # sys.argv[2]
    file_type = args.emb_file_type  # sys.argv[7]
    output_file = args.output_file  # sys.argv[3]
    number_of_shuffles = args.shuffles_num  # int(sys.argv[4])
    train_ratio = args.train_ratio  # sys.argv[4]
    classification_method = args.method  # sys.argv[6]

    if train_ratio == "large":
        training_ratios = [i for i in np.arange(0.1, 1, 0.1)]
    elif train_ratio == "all":
        training_ratios = [i for i in np.arange(0.01, 0.1, 0.01)] + [i for i in np.arange(0.1, 1, 0.1)]
    elif train_ratio == "custom":
        training_ratios = [0.1, 0.5, 0.9]
    elif train_ratio == "0.1":
        training_ratios = [0.1]
    elif train_ratio == "0.5":
        training_ratios = [0.5]
    elif train_ratio == "0.9":
        training_ratios = [0.9]
    else:
        raise ValueError("Invalid training ratio")

    results = evaluate(
        graph_path, embedding_file, number_of_shuffles, training_ratios, classification_method, file_type=file_type
    )

    print_results(results=results, shuffle_std=False, detailed=False)
    save_results(results, output_file, shuffle_std=False, detailed=False)
