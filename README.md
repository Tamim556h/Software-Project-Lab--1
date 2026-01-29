# Sentiment Analyzer 
The Emotion/Sentiment Analyzer is a C++-based machine learning project designed to automatically classify text into six emotion categories:
Joy, Sadness, Anger, Fear, Surprise, and Neutral.

The system analyzes both training data and real-time user input using three different machine learning algorithms:

Naive Bayes Classifier

Logistic Regression

The main goal is to compare the accuracy and performance of these three algorithms for emotion classification and provide users with clear predictions for any sentence they type.


how to run :
to compile => run the command in Sentyment_Analyzer directory 
g++ -std=c++11 -o bin/emotion_ditector src/*.cpp -I./include 
then :
./bin/emotion_detector