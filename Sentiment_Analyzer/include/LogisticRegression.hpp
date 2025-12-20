#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include <string>
#include <vector>
#include <map>

/**
 * @class LogisticRegression
 * @brief Multinomial Logistic Regression (Softmax) classifier
 * 
 * Implements multi-class logistic regression with stochastic gradient descent
 * for emotion classification from text features.
 */
class LogisticRegression {
private:
    std::vector<std::string> classes;
    std::map<std::string, std::vector<double>> weights;
    std::map<std::string, double> bias;
    double learningRate;
    int epochs;
    
    // Helper: sigmoid function
    double sigmoid(double x);
    
    // Helper: one-hot encode labels
    std::vector<std::vector<int>> oneHotEncode(const std::vector<std::string> &labels);
    
public:
    LogisticRegression(double lr = 0.01, int ep = 100);
    
    void trainFromVectors(const std::vector<std::vector<int>> &vectors, 
                          const std::vector<std::string> &labels);
    std::string predict(const std::vector<int> &vector);
    double accuracy(const std::vector<std::vector<int>> &vectors, 
                    const std::vector<std::string> &labels);
};

#endif
