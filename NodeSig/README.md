## NodeSig: Binary Node Embeddings via Random Walk Diffusion

#### Description
As the scale of networks increases, most of the widely used learning-based graph representation models also face computational challenges. While there is a recent effort toward designing algorithms that solely deal with scalability issues, most of them behave poorly in terms of accuracy on downstream tasks. In this paper, we aim to study models that balance the trade-off between efficiency and accuracy. In particular, we propose \textsc{\modelabbrv}, a  scalable model that computes binary node representations. \textsc{\modelabbrv} exploits random walk diffusion probabilities via stable random projections towards efficiently computing embeddings in the Hamming space. Our extensive experimental evaluation on various networks has demonstrated that the proposed model achieves a good balance between accuracy and efficiency compared to well-known baseline models on the node classification and link prediction tasks.

#### Compilation

**1.** You can compile the codes by typing the following commands:
```
cd build
cmake CMakeLists.txt
make all
```
#### Learning Representations

**2.** You can learn the representations of nodes by
```
./nodesig --edgefile EDGE_FILE --embfile EMB_FILE --walklen WALK_LENGTH 
```
**3.** To see the detailed parameter settings, you can use
```
./nodesig --help
```

#### References
A. Celikkanat, A. N. Papadopoulos and F. D. Malliaros, [NodeSig: Binary Node Embeddings via Random Walk Diffusion](.), The 2022 IEEE/ACM International Conference on Advances in Social Network Analysis and Mining, Istanbul, Turkey, 2022.


#### Notes
It might be required to install OpenMP library.
