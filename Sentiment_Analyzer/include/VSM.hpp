#ifndef VSM_HPP
#define VSM_HPP

#include <string>
#include <vector>
#include <map>
#include <cmath>

/**
 * @class VSM
 * @brief Vector Space Model with TF-IDF and Cosine Similarity
 * 
 * Implements TF-IDF vectorization and centroid-based classification
 * using cosine similarity for emotion detection.
 */
class VSM {
private:
    std::vector<std::string> classes;
    std::map<std::string, std::vector<double>> classCentroids;
    std::vector<std::vector<double>> trainVectors;
    std::vector<std::string> trainLabels;
    
    // Helper: compute cosine similarity between two vectors
    double cosineSimilarity(const std::vector<double> &a, const std::vector<double> &b);
    
    // Helper: compute TF-IDF vectors
    std::vector<std::vector<double>> computeTFIDF(const std::vector<std::vector<int>> &countVectors);
    
public:
    VSM();
    
    void trainFromVectors(const std::vector<std::vector<int>> &vectors, 
                          const std::vector<std::string> &labels);
    std::string predict(const std::vector<int> &vector);
    double accuracy(const std::vector<std::vector<int>> &vectors, 
                    const std::vector<std::string> &labels);
};

#endif
