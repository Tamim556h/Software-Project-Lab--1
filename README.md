# 🎭 Sentiment Analyzer

📌 Project Overview: 
The Sentiment Analyzer is a C++-based machine learning project designed to automatically classify text into six emotion categories:

😊 Joy  😢 Sadness  😠 Anger    😨 Fear 😲 Surprise 😐 Disgust

The system analyzes both:
📚 Training dataset
⌨️ Real-time user input

🧠 Machine Learning Algorithms Used 
This project implements and compares the following algorithms:
Naive Bayes Classifier
Logistic Regression

The main objective of this project is to:
Compare the accuracy of different ML algorithms
Evaluate performance differences
Provide clear predictions for user-input sentences


# 🏗️ Project Structure
Sentiment_Analyzer/
│
├── src/            # Source files (.cpp)
├── include/        # Header files
├── bin/            # Compiled output
└── dataset/        # Training data

# ⚙️ How to Compile
Navigate to the project root directory:
cd Sentiment_Analyzer
Then compile using:
g++ -std=c++11 -o bin/emotion_detector src/*.cpp -I./include

# ▶️ How to Run
After successful compilation:
./bin/emotion_detector

📊 Features
Text preprocessing
Feature extraction
Probability-based classification
Multi-class emotion detection
Accuracy comparison between models
