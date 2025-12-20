#include "../include/VSM.hpp"
#include <iostream>
#include <algorithm>

VSM::VSM() {
    classes.clear();
    classCentroids.clear();
    trainVectors.clear();
    trainLabels.clear();
}

double VSM::cosineSimilarity(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size()) return 0.0;
    
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    normA = std::sqrt(normA);
    normB = std::sqrt(normB);
    
    if (normA < 1e-10 || normB < 1e-10) return 0.0;
    
    return dotProduct / (normA * normB);
}

std::vector<std::vector<double>> VSM::computeTFIDF(const std::vector<std::vector<int>> &countVectors) {
    int numDocs = (int)countVectors.size();
    if (numDocs == 0) return std::vector<std::vector<double>>();
    
    int vocabSize = (int)countVectors[0].size();
    
    // Compute IDF for each term
    std::vector<int> docFreq(vocabSize, 0);
    for (int i = 0; i < numDocs; ++i) {
        for (int j = 0; j < vocabSize; ++j) {
            if (countVectors[i][j] > 0) {
                docFreq[j]++;
            }
        }
    }
    
    std::vector<double> idf(vocabSize);
    for (int j = 0; j < vocabSize; ++j) {
        if (docFreq[j] > 0) {
            idf[j] = std::log((double)numDocs / (double)docFreq[j]);
        } else {
            idf[j] = 0.0;
        }
    }
    
    // Compute TF-IDF vectors
    std::vector<std::vector<double>> tfidfVectors;
    for (int i = 0; i < numDocs; ++i) {
        std::vector<double> tfidfVec(vocabSize);
        double norm = 0.0;
        
        for (int j = 0; j < vocabSize; ++j) {
            double tf = (double)countVectors[i][j];
            tfidfVec[j] = tf * idf[j];
            norm += tfidfVec[j] * tfidfVec[j];
        }
        
        // L2 normalization
        norm = std::sqrt(norm);
        if (norm > 1e-10) {
            for (int j = 0; j < vocabSize; ++j) {
                tfidfVec[j] /= norm;
            }
        }
        
        tfidfVectors.push_back(tfidfVec);
    }
    
    return tfidfVectors;
}

void VSM::trainFromVectors(const std::vector<std::vector<int>> &vectors, 
                           const std::vector<std::string> &labels) {
    classes.clear();
    classCentroids.clear();
    
    // Compute TF-IDF vectors
    std::vector<std::vector<double>> tfidfVecs = computeTFIDF(vectors);
    trainVectors = tfidfVecs;
    trainLabels = labels;
    
    int numDocs = (int)vectors.size();
    if (numDocs == 0) return;
    
    int vecSize = (int)vectors[0].size();
    
    // Find unique classes
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
    
    // Compute centroids for each class
    for (size_t c = 0; c < classes.size(); ++c) {
        std::vector<double> centroid(vecSize, 0.0);
        int count = 0;
        
        for (int i = 0; i < numDocs; ++i) {
            if (labels[i] == classes[c]) {
                for (int j = 0; j < vecSize; ++j) {
                    centroid[j] += tfidfVecs[i][j];
                }
                count++;
            }
        }
        
        if (count > 0) {
            for (int j = 0; j < vecSize; ++j) {
                centroid[j] /= (double)count;
            }
        }
        
        classCentroids[classes[c]] = centroid;
    }
}

std::string VSM::predict(const std::vector<int> &vector) {
    // Convert count vector to TF-IDF
    std::vector<std::vector<int>> tempVec;
    tempVec.push_back(vector);
    std::vector<std::vector<double>> tfidfVec = computeTFIDF(tempVec);
    
    if (tfidfVec.empty() || tfidfVec[0].empty()) return "";
    
    double bestSim = -2.0;
    std::string bestClass = "";
    
    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        if (classCentroids.find(c) != classCentroids.end()) {
            double sim = cosineSimilarity(tfidfVec[0], classCentroids[c]);
            if (sim > bestSim) {
                bestSim = sim;
                bestClass = c;
            }
        }
    }
    
    if (bestClass == "" && classes.size() > 0) bestClass = classes[0];
    return bestClass;
}

double VSM::accuracy(const std::vector<std::vector<int>> &vectors, 
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
