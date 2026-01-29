#include "../include/Preprocessor.hpp"
#include <fstream>
#include <iostream>


Preprocessor::Preprocessor() {
    stopwords.clear();
}

// Basic character helpers (manual)
bool Preprocessor::is_space(char c) {

    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool Preprocessor::is_punct(char c) {
    // treat common punctuation as punctuation
    const char *punc = ".,!?;:'\"()[]{}<>-_/\\@#$%^&*+=|`~";
    for (int i = 0; punc[i] != '\0'; ++i) {
        if (c == punc[i]) return true;
    }
    return false;
}

char Preprocessor::to_lower_char(char c) {
    
    if (c >= 'A' && c <= 'Z') return c - 'A' + 'a';
    return c;
}

bool Preprocessor::equals_ignore_case(const std::string &a, const std::string &b) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {

        if (to_lower_char(a[i]) != to_lower_char(b[i])) return false;
    }

    return true;
}

bool Preprocessor::is_stopword(const std::string &w) {

    for (size_t i = 0; i < stopwords.size(); ++i) {
        if (equals_ignore_case(w, stopwords[i])) return true;
    }

    return false;
}

bool Preprocessor::is_negation_word(const std::string &w) {

    if (equals_ignore_case(w, "not")) return true;
    if (equals_ignore_case(w, "no")) return true;
    if (equals_ignore_case(w, "never")) return true;
    if (equals_ignore_case(w, "isn't")) return true;
    if (equals_ignore_case(w, "isnt")) return true;
    if (equals_ignore_case(w, "can't")) return true;
    if (equals_ignore_case(w, "cant")) return true;
    if (equals_ignore_case(w, "don't")) return true;
    if (equals_ignore_case(w, "dont")) return true;

    return false;
}


void Preprocessor::loadStopWords(const std::string &filePath) {
    stopwords.clear();
    std::ifstream infile(filePath.c_str());

    if (!infile.is_open()) {
        std::cerr << "Warning: could not open stopwords file: " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(infile, line))
    {
        // trim spaces from start and end manually
        size_t start = 0;
        while (start < line.size() && is_space(line[start])) start++;
        size_t end = line.size();

        while (end > start && is_space(line[end-1])) end--;
        if (end > start) {
            std::string w = line.substr(start, end - start);
            stopwords.push_back(w);
        }
    }

    infile.close();
}


// Main process: remove punctuation, lowercase, split on spaces, remove stopwords
std::vector<std::string> Preprocessor::process(const std::string &text) {
    std::vector<std::string> tokens;
    std::string word = "";
    bool negateNext = false;

    for (size_t i = 0; i < text.size(); ++i) {

        char c = text[i];
        c = to_lower_char(c);

        if (is_punct(c) || is_space(c)) {
            if (!word.empty()) {
                
                if (negateNext) {
                    std::string neg = "NOT_";
                    neg += word;
                    if (!is_stopword(neg)){
                        tokens.push_back(neg);
                    }
                    negateNext = false;
                } 
                else if (is_negation_word(word)) {
                    // set flag, do not output the negation token itself
                    negateNext = true;
                } 
                else {

                    if (!is_stopword(word)){
                        tokens.push_back(word);
                    }
                        
                }

                word.clear();
            }
            // skip punctuation/space
        } 
        else {
            // normal letter/digit
            word.push_back(c);
        }

    }

    // last word
    if (!word.empty()) {
        if (negateNext) {
            std::string neg = "NOT_";
            neg += word;
            if (!is_stopword(neg)){
                tokens.push_back(neg);
        } 
        else {
            if (!is_stopword(word)){
                tokens.push_back(word);
            }
                
        }
    }

    return tokens;
}
