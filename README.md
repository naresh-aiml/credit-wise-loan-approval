# CreditWise - Loan Approval Prediction

End-to-end ML pipeline to predict loan approval on imbalanced data.

## Models Compared

### 🏆 BEST - Naive Bayes (86.5%)
- Precision: 78.3% | Recall: 77.0% | F1: 77.6%
- Confusion Matrix: [[126,13],[14,47]]
- Why BEST? Best for imbalanced data - stable recall for defaulters

### 🥈 2nd BEST - Logistic Regression (87.5%)
- Precision: 79.0% | Recall: 80.3% | F1: 79.6%
- Confusion Matrix: [[126,13],[12,49]]
- Highest accuracy

### 🥉 3rd - KNN k=5 (75.5%)
- Precision: 62% | Recall: 50.8% | F1: 55.8%
- Confusion Matrix: [[120,19],[30,31]]

## Key Insights from EDA

- Credit_Score: +0.45 correlation (Higher score = Approval)
- DTI_Ratio: -0.44 correlation (Higher debt = Rejection)
- Data is imbalanced

## Pipeline

Applicant Data -> HashMap Encoding -> Scaling -> ML Models -> Approved/Rejected

## Tech Stack

Python, Scikit-Learn, Pandas, Jupyter

## Code

KNN:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

Naive Bayes:
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

Logistic Regression:
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

## What I Learned

In imbalanced data, Recall & F1 > Accuracy. Missing a defaulter is riskier.

## Author

[Naresh Kumar]
