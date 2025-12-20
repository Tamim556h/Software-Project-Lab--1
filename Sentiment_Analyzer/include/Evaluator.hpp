#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <string>
#include <vector>
#include <map>

/**
 * @class Evaluator
 * @brief Comprehensive metrics evaluation for multi-class classification
 * 
 * Computes accuracy, precision, recall, F1-score, and confusion matrices
 * for evaluating emotion detection model performance.
 */
class Evaluator {
private:
    std::vector<std::string> classes;
    std::map<std::string, std::map<std::string, int>> confusionMatrix;
    
    // Helper: check if class exists
    bool classExists(const std::string &c);
    
public:
    Evaluator();
    
    /**
     * Evaluate predictions against true labels
     * @param predictions Predicted emotion labels
     * @param trueLabels Ground truth emotion labels
     * @param classList List of all emotion classes
     */
    void evaluate(const std::vector<std::string> &predictions, 
                  const std::vector<std::string> &trueLabels,
                  const std::vector<std::string> &classList);
    
    // Get individual metrics
    double getAccuracy();
    double getPrecision(const std::string &className);
    double getRecall(const std::string &className);
    double getF1Score(const std::string &className);
    double getMacroF1();
    double getMicroF1();
    
    // Print comprehensive report
    void printReport();
    
    // Get confusion matrix
    std::map<std::string, std::map<std::string, int>> getConfusionMatrix();
};

#endif
