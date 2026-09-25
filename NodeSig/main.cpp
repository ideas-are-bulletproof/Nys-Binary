#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include "Graph.h"
#include "Model.h"
#include "Utilities.h"
#include <omp.h>

using namespace std;


template<typename T>
void normalizeRows(vector <vector <pair<unsigned int, T>>> &P);

template<typename T>
vector <vector <pair<unsigned int, T>>> constructTransitionMatrix(unsigned int numOfNodes, vector <vector <pair<unsigned int, double>>> adjList);

int main(int argc, char** argv) {

    typedef float T;
    string edgeFile, embFile, weightDistr;
    int numOfThreads;
    unsigned int walkLen, dimension, weightBlockSize, seed;
    bool directed, cyclicWeights, verbose;
    T alpha;

    // Default values
    directed = false;
    walkLen = 0;
    dimension = 8192;
    alpha = 1.0;
    weightDistr = "cauchy";
    numOfThreads = 0;
    cyclicWeights = false;
    weightBlockSize = 0;
    seed = 19;
    verbose = true;

    auto start_time = chrono::steady_clock::now();

    int err_code =  parse_arguments(argc, argv, edgeFile, embFile, walkLen, dimension, alpha, weightDistr, numOfThreads, cyclicWeights, weightBlockSize, seed, verbose);

    if(err_code != 0) {
        if(err_code < 0)
            cout << "+ Error code: " << err_code << endl;
        return 0;
    }

    if(numOfThreads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(numOfThreads);
    }

    if(verbose) {
        cout << "------------------------------------" << endl;
        cout << "+ Walk length: " << walkLen << endl;
        cout << "+ Dimension: " << dimension << endl;
        cout << "+ Alpha: " << alpha << endl;
        cout << "+ Weight distribution: " << weightDistr << endl;
        cout << "+ Number of threads: " << (numOfThreads ? to_string(numOfThreads) : to_string(omp_get_num_threads())
             + " (default max value)") << endl;
        cout << "+ Cyclic: " << cyclicWeights << endl;
        cout << "+ Weight block size: " << weightBlockSize << endl;
        cout << "+ Seed: " << seed << endl;
        cout << "------------------------------------" << endl;
    }

    if(numOfThreads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(numOfThreads);
    }

    Graph g = Graph(directed);
    g.readEdgeList(edgeFile, verbose);
    //g.writeEdgeList(dataset_path2, true);
    int unsigned numOfNodes = g.getNumOfNodes();

    // Get edge triplets
    vector <vector <pair<unsigned int, double>>> adjList = g.getAdjList<T>();
    vector <vector <pair<unsigned int, T>>> P = constructTransitionMatrix<T>(numOfNodes, adjList);

    Model<T> m(numOfNodes, dimension, weightDistr, cyclicWeights, seed, verbose);
    m.learnEmb(P, walkLen, alpha, weightBlockSize, embFile);

    auto end_time = chrono::steady_clock::now();
    if (verbose)
        cout << "+ Total elapsed time: " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count()
             << " seconds." << endl;

    return 0;

}

template<typename T>
void normalizeRows(vector <vector <pair<unsigned int, T>>> &P) {

    T rowSum;
    for(unsigned int node=0; node<P.size(); node++) {
        rowSum = 0.0;
        for(unsigned int nbIdx=0; nbIdx<P[node].size(); nbIdx++)
            rowSum += get<1>(P[node][nbIdx]);
        for(unsigned int nbIdx=0; nbIdx<P[node].size(); nbIdx++)
            get<1>(P[node][nbIdx]) = get<1>(P[node][nbIdx]) / rowSum;
    }

}

template<typename T>
vector <vector <pair<unsigned int, T>>> constructTransitionMatrix(unsigned int numOfNodes, vector <vector <pair<unsigned int, double>>> adjList) {

    vector <vector <pair<unsigned int, T>>> P;
    P.resize(numOfNodes);

    for(unsigned int node=0; node<numOfNodes; node++) {

        P[node].push_back(pair<unsigned int, double>(node, 1.0));

        for(unsigned int nbIdx=0; nbIdx<adjList[node].size(); nbIdx++) {
            pair <unsigned int, double> p = adjList[node][nbIdx];
            if(get<0>(p) > node) {
                P[node].push_back(p);
                P[get<0>(p)].push_back(pair<unsigned int, double>(node, get<1>(p)));
            }
        }
    }

    // Normalize row vectors
    normalizeRows(P);

    return P;
}
