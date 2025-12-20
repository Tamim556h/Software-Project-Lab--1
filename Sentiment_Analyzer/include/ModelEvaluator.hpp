#ifndef MODEL_EVALUATOR_HPP
#define MODEL_EVALUATOR_HPP

#include <string>
#include <vector>
#include <map>

/**
 * @class ModelEvaluator
 * @brief Advanced evaluation metrics for ML models
 * 
 * Computes confusion matrices, precision, recall, F1-scores,
 * and generates detailed per-emotion performance reports.
 */
class ModelEvaluator {
public:
    struct EvaluationMetrics {
        double accuracy;
        double macroAvgPrecision;
        double macroAvgRecall;
        double macroAvgF1;
        double microAvgPrecision;
        double microAvgRecall;
        double microAvgF1;
        std::map<std::string, std::map<std::string, double>> perClassMetrics;
        std::map<std::string, std::map<std::string, int>> confusionMatrix;
    };

    static EvaluationMetrics evaluate(
        const std::vector<std::string> &predictions,
        const std::vector<std::string> &actualLabels,
        const std::vector<std::string> &uniqueLabels
    );
    
    static void printDetailedReport(
        const std::string &algorithmName,
        const EvaluationMetrics &metrics
    );
    
    static void printConfusionMatrix(
        const std::map<std::string, std::map<std::string, int>> &matrix,
        const std::vector<std::string> &labels
    );

private:
    static double computeF1(double precision, double recall);
};

#endif
