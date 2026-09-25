# node2binary
Code and dataset used for the following paper:
```bash
NODE2BINARY: Compact Graph Node Embeddings Using Binary Vectors.
Niloy Talukder, Croix Gyurek and Mohammad Al Hasan.
Proceedings of the ACM Web Conference 2025 (WWW ’25).
```

## Running

The main syntax for our script is the following, which must be run from the directory containing `node2binary.py`.
```bash
python3 node2binary.py data/whatever_file  <dimension> <alpha> <beta> <gamma> <learn_rate> <learn_bias> <neg_samp_mult> <task>
```

Dimension, alpha, beta, gamma, and neg\_samp\_mult must be integers. Learn rate and bias can be floating point values. Task must be a string.


### Examples

To run Node Classification on PPI with 512-bit embeddings:
```bash
python3  node2binary.py data/PPI_edgelist --is_weighted --labels_path data/PPI_labels.txt --closed \
	512 25 10 1 0 0.008 0.01 8 1 --trees 1 --depth 3 \
	--task NodeClassification --verbose 0.01 \
	--iterations 1000 --stop-width 9999
```


To run Link Prediction on PPI with 512-bit embeddings:
```bash
python3 node2binary.py data/PPI_edgelist --is_weighted --closed \
	512 15 10 2 0 0.008 0.01 8 1 --trees 1 --depth 7 \
        --task LinkPrediction --verbose 0.01 \
	--iterations 800 --stop-width 9999
```
 

### Hyper-parameter explanations
`alpha` is the positive sample weight. `beta` is the negative sample weight. Note that the formula `(alpha * pos_grad + beta * neg_grad) * LR` is used, so doubling alpha and beta has exactly the same effect as doubling the LR.

`gamma` is the similarity weight, which aims to move sibling nodes (two nodes with the same direct parent) closer together. 

`neg_samp_mult` is the ratio of negative samples to positive samples during training.

Other options:
- `--is_weighted` this flag must be set if edgelist file has a weight column
- `--labels_path` path to data label file for Node Classification task
- `--subsample_ratio <number>` ratio of nodes to subsample from original dataset
- `--trees  <number>` number of hierarchy partition trees. Set to 1 as we used only one tree.
- `--depth <number>` number of levels in the tree.
- `--iterations <number>` stops after this number of iterations.
- `--testing_ratio <number>` ratio of test set used for evaluation
- `--binary_operator <choice>` binary operator used for Link Prediction
- `--seed <number>` seed for random number reproducibility
- `--stop-width <number>` controls the early-stop criterion. If, of the last `2*stop_width` iterations, the first `stop_width` have a higher average validation accuracy than the last `stop_width` iterations, the code stops.
- `--verbose <number>`: Number can be 2 or a real number between 0 and 1 inclusive. If 0, prints only at start and end of training. If 1, prints validation accuracy and training time for each iteration. If `x` strictly between 0 and 1, prints every `1.0/x` iterations. If 2, prints embedding and flipped bits each iteration.

#### Deprecated options:
- `neg_sim_weight` negative sample weight for sibling pairs. Negative samples for sibling pairs are not considered. Set to 0. 
- `neg_sim_multiplier` is the ratio of negative to positive sibling pairs during training. Set to 1.
- `--batchsize <number>` controls the batch size. **Defaults to the entire dataset if not present**.
- `--add_edgelist_to_siblings` if this flag is set then edgelist is added to the sibling pairs. **Not implemented**
- `--q` quality of the community partition tree. **Not implemented**

