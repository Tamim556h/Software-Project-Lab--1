#include "../include/ModelEvaluator.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

ModelEvaluator::EvaluationMetrics ModelEvaluator::evaluate(
    const std::vector<std::string> &predictions,
    const std::vector<std::string> &actualLabels,
    const std::vector<std::string> &uniqueLabels
) {
    EvaluationMetrics metrics;
    
    // Initialize confusion matrix
    for (const auto &label : uniqueLabels) {
        for (const auto &pred : uniqueLabels) {
            metrics.confusionMatrix[label][pred] = 0;
        }
    }
    
    // Build confusion matrix and calculate accuracy
    int correctCount = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        metrics.confusionMatrix[actualLabels[i]][predictions[i]]++;
        if (predictions[i] == actualLabels[i]) {
            correctCount++;
        }
    }
    
    metrics.accuracy = static_cast<double>(correctCount) / predictions.size();
    
    // Calculate per-class metrics
    double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
    int totalTP = 0, totalFP = 0, totalFN = 0;
    
    for (const auto &label : uniqueLabels) {
        int tp = metrics.confusionMatrix[label][label];
        int fp = 0, fn = 0;
        
        for (const auto &actualLabel : uniqueLabels) {
            if (actualLabel != label) {
                fp += metrics.confusionMatrix[actualLabel][label];
                fn += metrics.confusionMatrix[label][actualLabel];
            }
        }
        
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = computeF1(precision, recall);
        
        metrics.perClassMetrics[label]["precision"] = precision;
        metrics.perClassMetrics[label]["recall"] = recall;
        metrics.perClassMetrics[label]["f1"] = f1;
        metrics.perClassMetrics[label]["support"] = tp + fn;
        
        totalPrecision += precision;
        totalRecall += recall;
        totalF1 += f1;
        totalTP += tp;
        totalFP += fp;
        totalFN += fn;
    }
    
    // Macro averages
    metrics.macroAvgPrecision = totalPrecision / uniqueLabels.size();
    metrics.macroAvgRecall = totalRecall / uniqueLabels.size();
    metrics.macroAvgF1 = totalF1 / uniqueLabels.size();
    
    // Micro averages
    metrics.microAvgPrecision = static_cast<double>(totalTP) / (totalTP + totalFP);
    metrics.microAvgRecall = static_cast<double>(totalTP) / (totalTP + totalFN);
    metrics.microAvgF1 = computeF1(metrics.microAvgPrecision, metrics.microAvgRecall);
    
    return metrics;
}

double ModelEvaluator::computeF1(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void ModelEvaluator::printDetailedReport(
    const std::string &algorithmName,
    const EvaluationMetrics &metrics
) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ " << std::left << std::setw(66) << (algorithmName + " - DETAILED EVALUATION") << "║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣" << std::endl;
    
    // Overall metrics
    std::cout << "║ Overall Accuracy: " << std::fixed << std::setprecision(4) << std::setw(50) << metrics.accuracy << "║" << std::endl;
    std::cout << "║ Macro-Avg F1: " << std::fixed << std::setprecision(4) << std::setw(54) << metrics.macroAvgF1 << "║" << std::endl;
    std::cout << "║ Micro-Avg F1: " << std::fixed << std::setprecision(4) << std::setw(54) << metrics.microAvgF1 << "║" << std::endl;
    
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Per-Emotion Performance:                                          ║" << std::endl;
    std::cout << "╠══════════════╦══════════╦══════════╦══════════╦══════════════════╣" << std::endl;
    std::cout << "║ Emotion      ║ Precision║ Recall   ║ F1-Score ║ Support          ║" << std::endl;
    std::cout << "╠══════════════╬══════════╬══════════╬══════════╬══════════════════╣" << std::endl;
    
    for (const auto &pair : metrics.perClassMetrics) {
        std::cout << "║ " << std::left << std::setw(12) << pair.first 
                  << " ║ " << std::fixed << std::setprecision(6) << std::setw(8) << pair.second.at("precision")
                  << " ║ " << std::fixed << std::setprecision(6) << std::setw(8) << pair.second.at("recall")
                  << " ║ " << std::fixed << std::setprecision(6) << std::setw(8) << pair.second.at("f1")
                  << " ║ " << std::right << std::setw(16) << static_cast<int>(pair.second.at("support"))
                  << " ║" << std::endl;
    }
    
    std::cout << "╚══════════════╩══════════╩══════════╩══════════╩══════════════════╝" << std::endl;
}

void ModelEvaluator::printConfusionMatrix(
    const std::map<std::string, std::map<std::string, int>> &matrix,
    const std::vector<std::string> &labels
) {
    std::cout << "\nConfusion Matrix:" << std::endl;
    
    // Header
    std::cout << std::setw(12) << "Actual\\Pred";
    for (const auto &label : labels) {
        std::cout << std::setw(10) << label;
    }
    std::cout << std::endl;
    
    // Rows
    for (const auto &actualLabel : labels) {
        std::cout << std::setw(12) << actualLabel;
        for (const auto &predLabel : labels) {
            if (matrix.at(actualLabel).count(predLabel)) {
                std::cout << std::setw(10) << matrix.at(actualLabel).at(predLabel);
            } else {
                std::cout << std::setw(10) << 0;
            }
        }
        std::cout << std::endl;
    }
}
