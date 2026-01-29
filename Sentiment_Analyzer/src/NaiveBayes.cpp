#include "../include/NaiveBayes.hpp"
#include <cmath>
#include <iostream>


NaiveBayes::NaiveBayes() {
    classes.clear();
    classDocCount.clear();
    totalWordsInClass.clear();
    wordCountPerClass.clear();
    priorProb.clear();
    condProb.clear();
    vocabSize = 0;
}

bool NaiveBayes::classExists(const std::string &c) {
    for (size_t i = 0; i < classes.size(); ++i)
        if (classes[i] == c) return true;
    return false;
}

// Batch training using documents and vocabulary (computes priors and conditional probabilities)
void NaiveBayes::trainFromDocuments(const std::vector<std::vector<std::string>> &docs, 
                                    const std::vector<std::string> &labels, 
                                    const std::vector<std::string> &vocab) {
    // reset
    classes.clear();
    classDocCount.clear();
    totalWordsInClass.clear();
    wordCountPerClass.clear();
    priorProb.clear();
    condProb.clear();
    vocabSize = (int)vocab.size();

    int N = (int)docs.size();

    // gather classes
    for (int i = 0; i < N; ++i) {
        const std::string &lab = labels[i];

        if (!classExists(lab)){
            classes.push_back(lab);
        }
        if (classDocCount.find(lab) == classDocCount.end()) {
            classDocCount[lab] = 0;
        }

        // ensure maps exist
        if (totalWordsInClass.find(lab) == totalWordsInClass.end()) {
            totalWordsInClass[lab] = 0;
        }
        if (wordCountPerClass.find(lab) == wordCountPerClass.end()) {
            wordCountPerClass[lab] = std::map<std::string,int>();
        }
    }

    // count docs and words
    for (int i = 0; i < N; ++i) {

        const std::string &lab = labels[i];
        classDocCount[lab] += 1;
        const std::vector<std::string> &tokens = docs[i];

        for (size_t t = 0; t < tokens.size(); ++t) {
            const std::string &w = tokens[t];

            if (wordCountPerClass[lab].find(w) == wordCountPerClass[lab].end()) {
                wordCountPerClass[lab][w] = 1;
            }
            else {
                wordCountPerClass[lab][w] += 1;
            }
            totalWordsInClass[lab] += 1;
        }
    }

    // compute priors
    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        priorProb[c] = (double)classDocCount[c] / (double)N;
    }


    // compute conditional probabilities P(w|c) with Laplace smoothing
    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        condProb[c] = std::map<std::string, double>();

        // for all words in vocabulary compute (count+1)/(totalWordsInClass + vocabSize)
        for (int v = 0; v < vocabSize; ++v) {
            const std::string &w = vocab[v];
            int cnt = 0;
            if (wordCountPerClass[c].find(w) != wordCountPerClass[c].end()) {
                cnt = wordCountPerClass[c][w];
            }
            
            double prob = ((double)cnt + 1.0) / ((double)totalWordsInClass[c] + (double)vocabSize);
            condProb[c][w] = prob;

        }
    }
}

// Predict using log-probabilities
std::string NaiveBayes::predict(const std::vector<std::string> &tokens) {

    double bestScore = -INFINITY;
    std::string bestClass = "";

    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        double score = 0.0;
        // prior
        double pC = 1e-12;
        if (priorProb.find(c) != priorProb.end()) {
            pC = priorProb[c];
        }
        if (pC <= 0.0) {
            score = -INFINITY;
        }
        else {
            score = std::log(pC);
        }

        // add log-likelihoods
        for (size_t t = 0; t < tokens.size(); ++t) {
            const std::string &w = tokens[t];
            double pwc = 0.0;

            if (condProb[c].find(w) != condProb[c].end()) {
                pwc = condProb[c][w];
            }
            else {
                // word not in vocabulary or not seen for this class: Laplace smoothing with count=0
                double denom = (double)totalWordsInClass[c] + (double)vocabSize;

                if (denom <= 0.0) {
                    pwc = 1.0 / (double)(vocabSize + 1);
                }
                else {
                    pwc = 1.0 / denom;
                }
            }

            if (pwc > 0.0) {
                score += std::log(pwc);
            }
            else {
                score += -1e9;
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestClass = c;
        }
    }

    if (bestClass == "" && classes.size() > 0) bestClass = classes[0];

    
    return bestClass;
}

// compute accuracy on dataset
double NaiveBayes::accuracy(const std::vector<std::vector<std::string>> &docs, 
                            const std::vector<std::string> &labels) {
    int n = (int)docs.size();
    if (n == 0) return 0.0;
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        std::string pred = predict(docs[i]);
        if (pred == labels[i]) correct++;
    }
    return (double)correct / (double)n;
}
