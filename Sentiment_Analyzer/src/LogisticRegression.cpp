#include "../include/LogisticRegression.hpp"
#include <cmath>
#include <iostream>

LogisticRegression::LogisticRegression(double lr, int ep) : learningRate(lr), epochs(ep) {
    classes.clear();
    weights.clear();
    bias.clear();
}

double LogisticRegression::sigmoid(double x) {
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<std::vector<int>> LogisticRegression::oneHotEncode(const std::vector<std::string> &labels) {
    int n = (int)labels.size();
    int numClasses = (int)classes.size();
    
    std::vector<std::vector<int>> encoded(n, std::vector<int>(numClasses, 0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            if (labels[i] == classes[j]) {
                encoded[i][j] = 1;
            }
        }
    }
    
    return encoded;
}

void LogisticRegression::trainFromVectors(const std::vector<std::vector<int>> &vectors, 
                                          const std::vector<std::string> &labels) {
    int numDocs = (int)vectors.size();
    if (numDocs == 0) return;
    
    int vocabSize = (int)vectors[0].size();
    
    // Find unique classes
    classes.clear();
    for (int i = 0; i < numDocs; ++i) {
        const std::string &c = labels[i];
        bool found = false;
        for (size_t j = 0; j < classes.size(); ++j) {
            if (classes[j] == c) {
                found = true;
                break;
            }
        }
        if (!found) classes.push_back(c);
    }
    
    int numClasses = (int)classes.size();
    
    // Initialize weights and bias
    weights.clear();
    bias.clear();
    for (int c = 0; c < numClasses; ++c) {
        weights[classes[c]] = std::vector<double>(vocabSize, 0.0);
        bias[classes[c]] = 0.0;
    }
    
    // Convert to double vectors for training
    std::vector<std::vector<double>> doubleVectors(numDocs);
    for (int i = 0; i < numDocs; ++i) {
        doubleVectors[i].resize(vocabSize);
        for (int j = 0; j < vocabSize; ++j) {
            doubleVectors[i][j] = (double)vectors[i][j];
        }
    }
    
    // One-hot encode labels
    std::vector<std::vector<int>> yEncoded = oneHotEncode(labels);
    
    // Stochastic gradient descent
    for (int ep = 0; ep < epochs; ++ep) {
        for (int i = 0; i < numDocs; ++i) {
            // Forward pass
            std::vector<double> predictions(numClasses);
            for (int c = 0; c < numClasses; ++c) {
                double z = bias[classes[c]];
                for (int j = 0; j < vocabSize; ++j) {
                    z += weights[classes[c]][j] * doubleVectors[i][j];
                }
                predictions[c] = sigmoid(z);
            }
            
            // Backward pass (gradient descent)
            for (int c = 0; c < numClasses; ++c) {
                double error = predictions[c] - (double)yEncoded[i][c];
                
                // Update bias
                bias[classes[c]] -= learningRate * error;
                
                // Update weights
                for (int j = 0; j < vocabSize; ++j) {
                    double gradient = error * doubleVectors[i][j];
                    weights[classes[c]][j] -= learningRate * gradient;
                }
            }
        }
    }
}

std::string LogisticRegression::predict(const std::vector<int> &vector) {
    int vocabSize = (int)vector.size();
    
    double bestProb = -1.0;
    std::string bestClass = "";
    
    for (size_t c = 0; c < classes.size(); ++c) {
        const std::string &className = classes[c];
        
        double z = bias[className];
        for (int j = 0; j < vocabSize; ++j) {
            z += weights[className][j] * (double)vector[j];
        }
        
        double prob = sigmoid(z);
        if (prob > bestProb) {
            bestProb = prob;
            bestClass = className;
        }
    }
    
    if (bestClass == "" && classes.size() > 0) bestClass = classes[0];
    return bestClass;
}

double LogisticRegression::accuracy(const std::vector<std::vector<int>> &vectors, 
                                    const std::vector<std::string> &labels) {
    int n = (int)vectors.size();
    if (n == 0) return 0.0;
    
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        std::string pred = predict(vectors[i]);
        if (pred == labels[i]) correct++;
    }
    
    return (double)correct / (double)n;
}
