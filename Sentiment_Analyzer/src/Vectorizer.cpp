#include "../include/Vectorizer.hpp"


Vectorizer::Vectorizer() {
    vocabulary.clear();
}

int Vectorizer::find_in_vocab(const std::string &word) {

    for (size_t i = 0; i < vocabulary.size(); ++i) {
        if (vocabulary[i] == word) return (int)i;
    }
    return -1;
}

void Vectorizer::buildVocabulary(const std::vector<std::vector<std::string>> &documents) {

    vocabulary.clear();
    for (size_t i = 0; i < documents.size(); ++i) {
        const std::vector<std::string> &tokens = documents[i];

        for (size_t j = 0; j < tokens.size(); ++j) {
            const std::string &w = tokens[j];
            if (find_in_vocab(w) == -1) {
                vocabulary.push_back(w);
            }
        }
    }
}

// Create bag-of-words count vector for a single token list
std::vector<int> Vectorizer::transformSingle(const std::vector<std::string> &tokens) {
    std::vector<int> vec;
    vec.assign(vocabulary.size(), 0);

    for (size_t t = 0; t < tokens.size(); ++t) {
        int idx = find_in_vocab(tokens[t]);
        if (idx != -1) {
            vec[idx] += 1; 
        }
    }
    return vec;
}

// Transform multiple documents
std::vector<std::vector<int>> Vectorizer::transform(const std::vector<std::vector<std::string>> &documents) {
    std::vector<std::vector<int>> matrix;
    
    for (size_t i = 0; i < documents.size(); ++i) {
        matrix.push_back(transformSingle(documents[i]));
    }
    return matrix;
}

std::vector<std::string> Vectorizer::getVocabulary() {
    return vocabulary;
}
