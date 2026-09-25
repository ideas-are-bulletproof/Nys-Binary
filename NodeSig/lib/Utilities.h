#ifndef UTILITIES_H
#define UTILITIES_H
#include <string>
#include <sstream>
#include <vector>
#include <iostream>


namespace Constants
{
    const std::string ProgramName = "nodesig";
};

using namespace std;



int parse_arguments(int argc, char** argv, string &edgeFile, string &embFile, unsigned int &walkLen,
                    unsigned int &dimension, float &alpha, string &weightDistr,
                    int &numOfThreads, bool &cyclicWeights, unsigned int &weightBlockSize, unsigned int &seed,
                    bool &verbose) {

    vector <string> parameter_names{"--help",
                                    "--edgefile", "--embfile", "--walklen", "--dim",
                                    "--alpha", "--weightdistr", "--numthreads", "--cyclic", "--blocksize",
                                    "--seed", "--verbose"
    };

    string arg_name;
    stringstream help_msg, help_msg_required, help_msg_opt;

    // Set the help message
    help_msg_required << "\nUsage: ./" << Constants::ProgramName;
    help_msg_required << " " << parameter_names[1] << " EDGE_FILE "
                      << parameter_names[2] << " EMB_FILE "
                      << parameter_names[3] << " WALK_LENGTH "<< "\n";

    help_msg_opt << "\nOptional parameters:\n";
    help_msg_opt << "\t[ " << parameter_names[4] << " (Default: " << dimension << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[5] << " (Default: " << alpha << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[6] << " (Default: " << weightDistr << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[7] << " (Default: " << numOfThreads << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[8] << " (Default: " << cyclicWeights << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[9] << " (Default: " << weightBlockSize << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[10] << " (Default: " << seed << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[11] << " (Default: " << verbose << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[0] << ", -h ] Shows this message";

    help_msg << "" << help_msg_required.str() << help_msg_opt.str();

    // Read the argument values
    for(int i=1; i<argc; i=i+2) {

        arg_name.assign(argv[i]);

        if (arg_name.compare(parameter_names[1]) == 0) {
            edgeFile = argv[i + 1];
        } else if (arg_name.compare(parameter_names[2]) == 0) {
            embFile = argv[i + 1];
        } else if (arg_name.compare(parameter_names[3]) == 0) {
            walkLen = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[4]) == 0) {
            dimension = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[5]) == 0) {
            alpha = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[6]) == 0) {
            weightDistr = argv[i + 1];
        } else if (arg_name.compare(parameter_names[7]) == 0) {
            numOfThreads = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[8]) == 0) {
            cyclicWeights = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[9]) == 0) {
            weightBlockSize = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[10]) == 0) {
            seed = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[11]) == 0) {
            verbose = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[0]) == 0 or arg_name.compare("-h") == 0) {
            cout << help_msg.str() << endl;
            return 1;
        } else {
            cout << "Invalid argument name: " << arg_name << endl;
            return -2;
        }
        arg_name.clear();

    }

    // Check if the required parameters were set or not
    if(edgeFile.empty() || embFile.empty() || walkLen == 0) {
        cout << "Please enter the required parameters: ";
        cout << help_msg_required.str() << endl;

        return -4;
    }

    // Check if the constraints are satisfied
    if( alpha < 0 ) {
        cout << "alpha should be greater than 0." << endl;
        return -5;
    }
    if( weightDistr.compare("cauchy") != 0 && weightDistr.compare("gauss") != 0 ) {
        cout << "The distribution should be cauchy or gauss!" << weightDistr << endl;
        return -6;
    }
    if(numOfThreads < 0) {
        cout << "The number of threads must be greater than zero!" << endl;
        return -7;
    }

    if(seed < 0) {
        cout << "The seed must be an integer greater or equal to 0!" << endl;
        return -8;
    }

    if(cyclicWeights != 1 && cyclicWeights != 0) {
        cout << "The cyclic must be 1 or 0!" << endl;
        return -9;
    }

    return 0;

}

#endif //UTILITIES_H