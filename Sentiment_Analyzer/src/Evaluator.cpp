#include "../include/Evaluator.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

Evaluator::Evaluator() {
    classes.clear();
    confusionMatrix.clear();
}

bool Evaluator::classExists(const std::string &c) {
    for (size_t i = 0; i < classes.size(); ++i) {
        if (classes[i] == c) return true;
    }
    return false;
}

void Evaluator::evaluate(const std::vector<std::string> &predictions,
                         const std::vector<std::string> &trueLabels,
                         const std::vector<std::string> &classList) {
    classes = classList;
    confusionMatrix.clear();
    
    // Initialize confusion matrix
    for (size_t i = 0; i < classes.size(); ++i) {
        confusionMatrix[classes[i]] = std::map<std::string, int>();
        for (size_t j = 0; j < classes.size(); ++j) {
            confusionMatrix[classes[i]][classes[j]] = 0;
        }
    }
    
    // Populate confusion matrix
    if (predictions.size() != trueLabels.size()) return;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        confusionMatrix[trueLabels[i]][predictions[i]]++;
    }
}

double Evaluator::getAccuracy() {
    int correct = 0, total = 0;
    
    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        for (size_t j = 0; j < classes.size(); ++j) {
            const std::string &c2 = classes[j];
            total += confusionMatrix[c][c2];
            if (c == c2) correct += confusionMatrix[c][c2];
        }
    }
    
    return total == 0 ? 0.0 : (double)correct / (double)total;
}

double Evaluator::getPrecision(const std::string &className) {
    int tp = confusionMatrix[className][className];
    int fp = 0;
    
    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        if (c != className) {
            fp += confusionMatrix[c][className];
        }
    }
    
    return (tp + fp) == 0 ? 0.0 : (double)tp / (double)(tp + fp);
}

double Evaluator::getRecall(const std::string &className) {
    int tp = confusionMatrix[className][className];
    int fn = 0;
    
    for (size_t j = 0; j < classes.size(); ++j) {
        const std::string &c = classes[j];
        if (c != className) {
            fn += confusionMatrix[className][c];
        }
    }
    
    return (tp + fn) == 0 ? 0.0 : (double)tp / (double)(tp + fn);
}

double Evaluator::getF1Score(const std::string &className) {
    double prec = getPrecision(className);
    double rec = getRecall(className);
    
    if (prec + rec < 1e-10) return 0.0;
    return 2.0 * (prec * rec) / (prec + rec);
}

double Evaluator::getMacroF1() {
    double sum = 0.0;
    for (size_t i = 0; i < classes.size(); ++i) {
        sum += getF1Score(classes[i]);
    }
    return classes.empty() ? 0.0 : sum / (double)classes.size();
}

double Evaluator::getMicroF1() {
    // For multi-class, macro F1 is typically used, but for completeness:
    return getAccuracy(); // Micro F1 equals accuracy for multi-class
}

void Evaluator::printReport() {
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         DETAILED EVALUATION REPORT                     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\nAccuracy: " << std::fixed << std::setprecision(4) << (getAccuracy() * 100.0) << "%\n" << std::endl;
    
    std::cout << std::left << std::setw(15) << "Emotion"
              << std::setw(12) << "Precision"
              << std::setw(12) << "Recall"
              << std::setw(12) << "F1-Score" << std::endl;
    std::cout << std::string(51, '-') << std::endl;
    
    for (size_t i = 0; i < classes.size(); ++i) {
        const std::string &c = classes[i];
        std::cout << std::left << std::setw(15) << c
                  << std::setw(12) << std::fixed << std::setprecision(4) << (getPrecision(c) * 100.0) << "%"
                  << std::setw(12) << std::fixed << std::setprecision(4) << (getRecall(c) * 100.0) << "%"
                  << std::setw(12) << std::fixed << std::setprecision(4) << (getF1Score(c) * 100.0) << "%" << std::endl;
    }
    
    std::cout << std::string(51, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "Macro Average"
              << std::setw(12) << "--"
              << std::setw(12) << "--"
              << std::setw(12) << std::fixed << std::setprecision(4) << (getMacroF1() * 100.0) << "%" << std::endl;
}

std::map<std::string, std::map<std::string, int>> Evaluator::getConfusionMatrix() {
    return confusionMatrix;
}
