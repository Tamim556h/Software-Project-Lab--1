#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cctype>
#include <algorithm>

#include "../include/Preprocessor.hpp"
#include "../include/Vectorizer.hpp"
#include "../include/NaiveBayes.hpp"
#include "../include/VSM.hpp"
#include "../include/LogisticRegression.hpp"
#include "../include/ModelEvaluator.hpp"

// Simple CSV loader: expects header line, then each line text,label
void loadCSV(const std::string &path, std::vector<std::string> &texts, std::vector<std::string> &labels) {
    texts.clear();
    labels.clear();
    std::ifstream infile(path.c_str());

    if (!infile.is_open()) {
        std::cerr << "Error: could not open file: " << path << std::endl;
        return;
    }

    std::string line;
    bool first = true;

    while (std::getline(infile, line)) 
    {
        if (first) { 
            first = false; continue; // skip header
        } 

        int pos = -1;
        for (int i = (int)line.size() - 1; i >= 0; --i) {
            if (line[i] == ',') { 
                pos = i; break; 
            }
        }
        if (pos == -1) continue;

        std::string text = line.substr(0, pos);
        std::string label = line.substr(pos + 1);
        
        // trim spaces from label
        size_t s = 0;
        while (s < label.size() && (label[s] == ' ' || label[s] == '\t' || label[s] == '\r' || label[s] == '\n')) s++;
        size_t e = label.size();

        while (e > s && (label[e-1] == ' ' || label[e-1] == '\t' || label[e-1] == '\r' || label[e-1] == '\n')) e--;
        if (e > s) label = label.substr(s, e - s);
        else label = "";

        texts.push_back(text);
        labels.push_back(label);
    }
    infile.close();
}

bool isValidInput(const std::string &input) {
    
    if (input.empty()) return false;
    
    // Check if input has at least one letter
    bool hasLetter = false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::isalpha(input[i])) {
            hasLetter = true;
            break;
        }
    }
    
    // Reject if only numbers or special characters
    if (!hasLetter) return false;
    
    return true;
}

void displayMainMenu() {

    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          EMOTION DETECTOR - MAIN MENU                 ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ 1. Train and Evaluate All Models                      ║" << std::endl;
    std::cout << "║ 2. Predict Emotion from User Input                    ║" << std::endl;
    std::cout << "║ 3. View Detailed Performance Report                   ║" << std::endl;
    std::cout << "║ 4. Exit                                               ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "Select option (1-4): ";

}

// Global model variables and trained data =>

NaiveBayes g_nb;
VSM g_vsm;
LogisticRegression g_lr(0.01, 100);
Vectorizer g_vec;
Preprocessor g_pre;
bool g_trained = false;

ModelEvaluator::EvaluationMetrics g_nbMetrics, g_vsmMetrics, g_lrMetrics;
std::vector<std::string> g_uniqueLabels;

void trainModels(const std::vector<std::string> &rawTexts, const std::vector<std::string> &labels) {
    if (rawTexts.empty()) {
        std::cerr << "Error: No training data loaded.\n";
        return;
    }

    g_uniqueLabels.clear();
    for (size_t i = 0; i < labels.size(); ++i) {
        if (std::find(g_uniqueLabels.begin(), g_uniqueLabels.end(), labels[i]) == g_uniqueLabels.end()) {
            g_uniqueLabels.push_back(labels[i]);
        }
    }
    std::cout << "\n[INFO] Unique emotions detected: " << g_uniqueLabels.size() << std::endl;

    for (const auto &emotion : g_uniqueLabels) {
        std::cout << "  - " << emotion << std::endl;
    }

    // Tokenize all documents
    std::vector<std::vector<std::string>> docs;
    for (size_t i = 0; i < rawTexts.size(); ++i) {
        std::vector<std::string> toks = g_pre.process(rawTexts[i]);
        docs.push_back(toks);
    }

    // Build vocabulary
    g_vec.buildVocabulary(docs);
    std::vector<std::string> vocab = g_vec.getVocabulary();
    
    std::cout << "[INFO] Vocabulary size: " << vocab.size() << " unique words\n" << std::endl;
    
    std::vector<std::vector<int>> countVectors = g_vec.transform(docs);

    std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        TRAINING ALL THREE ALGORITHMS                  ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════╣" << std::endl;

    
    // Train Naive Bayes
    std::cout << "║ 1. Training Naive Bayes...                            ║" << std::endl;
    g_nb.trainFromDocuments(docs, labels, vocab);
    std::vector<std::string> nbPredictions;

    for (size_t i = 0; i < docs.size(); ++i) {
        nbPredictions.push_back(g_nb.predict(docs[i]));
    }
    g_nbMetrics = ModelEvaluator::evaluate(nbPredictions, labels, g_uniqueLabels);
    double nbAcc = g_nbMetrics.accuracy;
    std::cout << "║    Accuracy: " << std::fixed << std::setprecision(2) << std::setw(38) << (nbAcc * 100.0) << "%   ║" << std::endl;


    // Train Vector Space Model (VSM)
    std::cout << "║ 2. Training Vector Space Model (VSM)...               ║" << std::endl;
    g_vsm.trainFromVectors(countVectors, labels);
    std::vector<std::string> vsmPredictions;

    for (size_t i = 0; i < countVectors.size(); ++i) {
        vsmPredictions.push_back(g_vsm.predict(countVectors[i]));
    }
    g_vsmMetrics = ModelEvaluator::evaluate(vsmPredictions, labels, g_uniqueLabels);
    double vsmAcc = g_vsmMetrics.accuracy;
    std::cout << "║    Accuracy: " << std::fixed << std::setprecision(2) << std::setw(38) << (vsmAcc * 100.0) << "%   ║" << std::endl;


    // Train Logistic Regression
    std::cout << "║ 3. Training Logistic Regression...                    ║" << std::endl;
    g_lr.trainFromVectors(countVectors, labels);
    std::vector<std::string> lrPredictions;

    for (size_t i = 0; i < countVectors.size(); ++i) {
        lrPredictions.push_back(g_lr.predict(countVectors[i]));
    }
    g_lrMetrics = ModelEvaluator::evaluate(lrPredictions, labels, g_uniqueLabels);
    double lrAcc = g_lrMetrics.accuracy;
    std::cout << "║    Accuracy: " << std::fixed << std::setprecision(2) << std::setw(38) << (lrAcc * 100.0) << "%   ║" << std::endl;
    
    std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;

    

    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║            ACCURACY COMPARISON TABLE                  ║" << std::endl;
    std::cout << "╠════════════════════════════╦═════════════════════════╣" << std::endl;
    std::cout << "║ Algorithm                  ║ Training Accuracy       ║" << std::endl;
    std::cout << "╠════════════════════════════╬═════════════════════════╣" << std::endl;
    std::cout << "║ Naive Bayes                ║ " << std::fixed << std::setprecision(2) << std::setw(19) << (nbAcc * 100.0) << "% ║" << std::endl;
    std::cout << "║ Vector Space Model (VSM)   ║ " << std::fixed << std::setprecision(2) << std::setw(19) << (vsmAcc * 100.0) << "% ║" << std::endl;
    std::cout << "║ Logistic Regression        ║ " << std::fixed << std::setprecision(2) << std::setw(19) << (lrAcc * 100.0) << "% ║" << std::endl;
    std::cout << "╚════════════════════════════╩═════════════════════════╝" << std::endl;

    g_trained = true;
}

void predictEmotion() {

    if (!g_trained) {
        std::cout << "\n[ERROR] Models not trained yet. Please train models first (option 1).\n";
        return;
    }

    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         EMOTION PREDICTION - INTERACTIVE MODE         ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Enter sentences to classify emotions.                 ║" << std::endl;
    std::cout << "║ (Type 'back' to return to main menu)                  ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝\n" << std::endl;
    
    while (true) {
        std::cout << "> Enter text: ";
        std::string input;
        if (!std::getline(std::cin, input)) break;

        // Check if user wants to exit
        std::string lower;
        for (size_t i = 0; i < input.size(); ++i) {
            lower.push_back(std::tolower(input[i]));
        }
        if (lower == "back" || lower == "quit" || lower == "exit") break;

        if (!isValidInput(input)) {
            std::cout << "[ERROR] Invalid input! Please enter a valid sentence with letters.\n" << std::endl;
            continue;
        }

        // Process text
        std::vector<std::string> tokens = g_pre.process(input);
        
        if (tokens.empty()) {
            std::cout << "[WARNING] No meaningful tokens found. Try a different sentence.\n" << std::endl;
            continue;
        }

        // Get predictions from all three models
        std::string nbPred = g_nb.predict(tokens);
        
        std::vector<int> countVec = g_vec.transformSingle(tokens);
        std::string vsmPred = g_vsm.predict(countVec);
        std::string lrPred = g_lr.predict(countVec);

        
        std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                 EMOTION PREDICTIONS                   ║" << std::endl;
        std::cout << "╠════════════════════════════╦═════════════════════════╣" << std::endl;
        std::cout << "║ Algorithm                  ║ Predicted Emotion       ║" << std::endl;
        std::cout << "╠════════════════════════════╬═════════════════════════╣" << std::endl;
        std::cout << "║ Naive Bayes                ║ " << std::left << std::setw(21) << nbPred << " ║" << std::endl;
        //std::cout << "║ Vector Space Model (VSM)   ║ " << std::left << std::setw(21) << vsmPred << " ║" << std::endl;
        std::cout << "║ Logistic Regression        ║ " << std::left << std::setw(21) << lrPred << " ║" << std::endl;
        std::cout << "╚════════════════════════════╩═════════════════════════╝\n" << std::endl;
    }
}

int main() {
    
    std::string dataPath = "data/dataset.csv";
    std::string stopPath = "data/stopwords.csv";

    // Load data once
    std::vector<std::string> rawTexts, labels;
    loadCSV(dataPath, rawTexts, labels);

    if (rawTexts.size() == 0) {
        std::cerr << "[ERROR] No data loaded. Ensure " << dataPath << " exists.\n";
        return 1;
    }

    std::cout << "[INFO] Loaded " << rawTexts.size() << " training samples from " << dataPath << std::endl;

    // Preprocess
    g_pre.loadStopWords(stopPath);

    while (true) {
        displayMainMenu();
        std::string choice;

        if (!std::getline(std::cin, choice)) break;

        if (choice == "1") {
            trainModels(rawTexts, labels);
        } 
        else if (choice == "2") {
            predictEmotion();
        } 
        else if (choice == "3") {
            if (!g_trained) {
                std::cout << "\n[ERROR] Models not trained yet. Please train models first (option 1).\n";
            } 
            else {
                std::cout << "\n";
                ModelEvaluator::printDetailedReport("NAIVE BAYES", g_nbMetrics);
                ModelEvaluator::printDetailedReport("VECTOR SPACE MODEL (VSM)", g_vsmMetrics);
                ModelEvaluator::printDetailedReport("LOGISTIC REGRESSION", g_lrMetrics);
            }
        }
        else if (choice == "4") {
            std::cout << "\nThank you for using EmotionDet!\n";
            break;
        } 
        else {
            std::cout << "[ERROR] Invalid option. Please select 1-4.\n";
        }

    }

    return 0;
}
