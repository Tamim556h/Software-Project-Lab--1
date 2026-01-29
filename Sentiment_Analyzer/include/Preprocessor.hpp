#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include <string>
#include <vector>

/**
 * @class Preprocessor
 * @brief Text preprocessing for emotion detection
 */

class Preprocessor {
private:
    std::vector<std::string> stopwords;

    // helper utilities implemented manually
    bool is_space(char c);
    bool is_punct(char c);
    char to_lower_char(char c);
    bool equals_ignore_case(const std::string &a, const std::string &b);
    bool is_stopword(const std::string &w);
    bool is_negation_word(const std::string &w);
    
public:
    Preprocessor();
    void loadStopWords(const std::string &filePath); // loads stopwords from file (one per line)
    std::vector<std::string> process(const std::string &text); // tokenize + lowercase + remove stopwords
    
    int getVocabularySize() const;
    int getStopwordCount() const;
};

#endif
