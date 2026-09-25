import numpy as np
import random
import networkx as nx
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import copy

def get_y_pred(y_test, y_pred_prob):
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred

def NodeClassification(embedding_look_up, node_list, labels, testing_ratio, subsample_ratio, seed=0):
    n_splits = 10
    # print("node_list:", node_list)
    # print("node_list size:", len(node_list))
    # print("labels:", labels)
    ss = ShuffleSplit(n_splits=n_splits, test_size=testing_ratio, random_state=seed)

    # Initialize scores
    accuracy_scores = []
    micro_f1_scores = []
    macro_f1_scores = []

    ## subsample from the nodelist
    random.seed(0)
    subsample_indices = random.sample(range(len(node_list)), int(len(node_list) * subsample_ratio))
    subsample_node_list = [node_list[i] for i in subsample_indices]
    
    # Initialize binarizer
    binarizer = MultiLabelBinarizer(sparse_output=True)
    all_labels = [labels[node] for node in subsample_node_list]
    all_labels = np.array(all_labels)
    binarizer.fit(all_labels)

    # Perform the splits and training/testing
    for train_index, test_index in ss.split(subsample_node_list):
        X_train = [embedding_look_up[subsample_node_list[i]] for i in train_index]
        X_train = np.asarray(X_train, dtype='f') # Convert to single precision floating point number
        y_train = binarizer.transform([labels[subsample_node_list[i]] for i in train_index]).todense()
        y_train = np.asarray(y_train)  # Convert to np.array
        X_test = [embedding_look_up[subsample_node_list[i]] for i in test_index]
        X_test = np.asarray(X_test, dtype='f') 
        y_test = binarizer.transform([labels[subsample_node_list[i]] for i in test_index]).todense()
        y_test = np.asarray(y_test)
        model = OneVsRestClassifier(LogisticRegression(random_state=seed, solver='liblinear', max_iter= 1000))    
        model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)

        # Use a small trick: assume we know how many labels to predict
        y_pred = get_y_pred(y_test, y_pred_prob)

        # Store the scores
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        micro_f1_scores.append(f1_score(y_test, y_pred, average="micro"))
        macro_f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    # Print and return the average performance
    avg_accuracy = np.mean(accuracy_scores)
    avg_micro_f1 = np.mean(micro_f1_scores)
    avg_macro_f1 = np.mean(macro_f1_scores)

    return avg_accuracy, avg_micro_f1, avg_macro_f1
