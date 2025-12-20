#ifndef VECTORIZER_HPP
#define VECTORIZER_HPP

#include <string>
#include <vector>

/**
 * @class Vectorizer
 * @brief Bag-of-words vectorization for text classification
 * 
 * Converts tokenized documents into numerical feature vectors
 * using vocabulary-based count representation.
 */
class Vectorizer {
private:
    std::vector<std::string> vocabulary; // list of unique words

    // helper: find index of word in vocabulary (-1 if not found)
    int find_in_vocab(const std::string &word);

public:
    Vectorizer();
    void buildVocabulary(const std::vector<std::vector<std::string>> &documents);
    std::vector<int> transformSingle(const std::vector<std::string> &tokens); // bag-of-words counts
    std::vector<std::vector<int>> transform(const std::vector<std::vector<std::string>> &documents);
    std::vector<std::string> getVocabulary();
};

#endif
