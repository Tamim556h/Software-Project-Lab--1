#ifndef NAIVEBAYES_HPP
#define NAIVEBAYES_HPP

#include <string>
#include <vector>
#include <map>

/**
 * @class NaiveBayes
 * @brief Multinomial Naive Bayes classifier for emotion detection
 * 
 * Implements Naive Bayes with Laplace smoothing for text classification.
 * Assumes independence between features (bag-of-words assumption).
 */

class NaiveBayes {
private:
    std::vector<std::string> classes; // list of emotion labels
    std::map<std::string, int> classDocCount;      // number of documents per class
    std::map<std::string, int> totalWordsInClass;  // total word counts per class
    std::map<std::string, std::map<std::string, int> > wordCountPerClass; // counts of each word per class
    std::map<std::string, double> priorProb;       // prior P(class)
    std::map<std::string, std::map<std::string, double> > condProb; // P(word|class) (with Laplace)
    int vocabSize;

    // helper: check if class exists in classes vector
    bool classExists(const std::string &c);

public:
    NaiveBayes();
    void trainFromDocuments(const std::vector<std::vector<std::string>> &docs, 
                            const std::vector<std::string> &labels, 
                            const std::vector<std::string> &vocab);
    std::string predict(const std::vector<std::string> &tokens);
    
    double accuracy(const std::vector<std::vector<std::string>> &docs, 
                    const std::vector<std::string> &labels);
};

#endif
