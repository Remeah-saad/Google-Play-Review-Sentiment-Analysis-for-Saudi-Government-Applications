Google Play Review Sentiment Analysis for Saudi Government Applications
Overview
This project leverages Machine Learning and Natural Language Processing (NLP) to analyze user feedback on Saudi government applications available on the Google Play Store. The system automatically classifies reviews as Positive or Negative, providing actionable insights for government entities to improve user satisfaction and resolve recurring technical issues.

Problem Statement
Government applications receive thousands of reviews. Manual analysis is inefficient and time-consuming. Our system automates this process to:

Detect recurring user issues.

Gauge overall user satisfaction.

Provide recommendations for application enhancements.

Project Objectives
Sentiment Classification: Accurately classify Arabic reviews into positive and negative sentiments.

Actionable Insights: Assist government entities in identifying pain points to improve app performance.

Dataset
Source: Google Play Store.

Content: Arabic reviews for Saudi government and healthcare applications.

Size: ~4,965 reviews.

Distribution: 52% Positive, 48% Negative (Balanced dataset).

Data Preprocessing Pipeline
To ensure high model accuracy, we implemented a rigorous preprocessing pipeline:

Filtering: Removed neutral reviews (3-star ratings) to focus on binary sentiment classification.

Cleaning: Stripped emojis, special symbols, numbers, and Latin characters.

Normalization: Unified Arabic character forms (e.g., converting 'أ', 'إ', 'آ' to 'ا').

Diacritics Removal: Stripped Arabic diacritics (Tashkeel) to match common user input patterns.

Data Splitting: Split into 70% Training, 15% Validation, and 15% Testing.

Feature Engineering & Modeling
Vectorization: Used TF-IDF to convert text data into numerical features, assigning higher weights to significant terms.

Models Evaluated: * Logistic Regression

Support Vector Machine (SVM)

Results
The SVM model outperformed Logistic Regression, achieving:

Accuracy: 81.5%

ROC-AUC: ~0.89

SVM demonstrated superior capability in distinguishing sentiment, with significantly lower misclassification rates compared to Logistic Regression.

Challenges
Sarcasm: Detecting negative sentiments expressed through positive wording (e.g., "The app is great, it crashes every second!").

Linguistic Complexity: Handling Arabic morphology, diverse dialects, and complex sentence structures.

Mixed Sentiments: Managing reviews that contain both praise and complaints.

Data Bias: Feedback is often polarized between highly satisfied or highly frustrated users.

Tech Stack
Language: Python

Libraries: Scikit-learn, google_play_scraper

Techniques: NLP, TF-IDF, SVM, Logistic Regression

Conclusion
This project demonstrates that Machine Learning, specifically SVM, is highly effective for Arabic sentiment analysis in a governmental context. It provides a scalable solution for monitoring user satisfaction and driving data-informed improvements for public services.
