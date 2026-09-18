# text_classification_imdb_sentiment_analysis.ipynb
IMDB movie review sentiment classification using NLP, TF-IDF, Logistic Regression, and Naive Bayes

# IMDb Movie Review Sentiment Analysis

## Project Overview

This project builds an end-to-end Natural Language Processing (NLP) machine learning pipeline to classify IMDb movie reviews as positive or negative.

The analysis includes text preprocessing, exploratory data analysis, TF-IDF feature extraction, machine learning model training, model comparison, feature interpretation, and error analysis.

## Dataset

The original dataset contains 50,000 IMDb movie reviews labeled as either positive or negative.

After removing 418 duplicate reviews, 49,582 reviews remained for analysis.

- Training reviews: 39,665
- Testing reviews: 9,917
- Positive reviews: 24,884
- Negative reviews: 24,698

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes

## Machine Learning Workflow

1. Loaded and inspected 50,000 IMDb movie reviews
2. Removed duplicate records
3. Cleaned and standardized review text
4. Explored review length and sentiment distribution
5. Converted sentiment labels into binary values
6. Split the data into training and testing sets
7. Generated 10,000 TF-IDF text features
8. Trained Logistic Regression and Multinomial Naive Bayes models
9. Evaluated models using accuracy, precision, recall, F1 score, and confusion matrices
10. Analyzed misclassified reviews and model errors
11. Tested the final model on previously unseen review examples

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.18% | 88.17% | 90.60% | 89.37% |
| Multinomial Naive Bayes | 86.20% | 85.12% | 87.84% | 86.46% |

Logistic Regression achieved higher held-out test performance across all four reported metrics.

## Error Analysis

The Logistic Regression model misclassified 1,073 of 9,917 test reviews, resulting in a 10.82% misclassification rate.

- False positives: 605
- False negatives: 468
- Correctly classified negative reviews: 4,335
- Correctly classified positive reviews: 4,509

## Key Findings

- Logistic Regression achieved 89.18% test accuracy and an 89.37% F1 score.
- Logistic Regression performed better than Multinomial Naive Bayes across the reported evaluation metrics.
- The dataset remained nearly evenly balanced after duplicate removal.
- Error analysis showed that 56.38% of the model's errors were false positives and 43.62% were false negatives.
- The trained pipeline successfully demonstrated sentiment predictions on new example movie reviews.

## Project Files

`text_classification_sentiment_analysis.ipynb` — Complete analysis, preprocessing, visualizations, machine learning models, evaluation, and error analysis.

## Skills Demonstrated

Python • Pandas • NumPy • Matplotlib • Scikit-learn • NLP • Text Preprocessing • TF-IDF • Logistic Regression • Naive Bayes • Classification Metrics • Confusion Matrices • Model Comparison • Error Analysis
