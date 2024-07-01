# Iris-Flower-Classification
## Abstract

This project focuses on the classification of the Iris dataset using various machine learning algorithms. The Iris dataset, a well-known dataset in the machine learning community, contains measurements of sepal length, sepal width, petal length, and petal width for three species of Iris flowers: setosa, versicolor, and virginica. The project begins with data preprocessing, including the removal of unnecessary columns and an analysis of class distributions. Exploratory data analysis is conducted using visualization techniques such as box plots, histograms, and pair plots to understand the relationships between features. The dataset is then split into training and testing sets. Multiple classification algorithms, including Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Naive Bayes, are applied to the data. The performance of each model is evaluated using metrics such as accuracy score and confusion matrix. This project demonstrates the effectiveness of different machine learning algorithms in solving a classic classification problem and provides insights into their comparative performance.
## Table of Contents
    - Introduction
    - Libraries Used
    - Dataset
    - Data Preprocessing
    - Exploratory Data Analysis
    - Model Training and Evaluation
        - Logistic Regression
        - Support Vector Machine
        - K-Nearest Neighbors
        - Naive Bayes
    Results
    Conclusion

## Introduction

The Iris dataset is one of the most famous datasets in the field of machine learning. It contains 150 observations of iris flowers, with measurements of sepal length, sepal width, petal length, and petal width, along with the species of each flower. The goal of this project is to classify iris flowers into one of the three species based on these measurements using various machine learning algorithms.
Libraries Used

    - NumPy
    - Matplotlib
    - Seaborn
    - Pandas
    - Scikit-learn

## Dataset

The dataset used in this project is the Iris dataset, which contains the following columns:

    - Unnamed: 0
    - Sepal.Length
    - Sepal.Width
    - Petal.Length
    - Petal.Width
    - Species

## Data Preprocessing

The dataset is initially examined, and unnecessary columns are dropped. The class distributions are analyzed to ensure balanced classes.
Exploratory Data Analysis

The distribution of different species in the dataset is analyzed using value counts. Various visualization techniques, including box plots, histograms, and pair plots, are used to understand the relationships between features and check for outliers.
Model Training and Evaluation
Logistic Regression

The Logistic Regression model is trained on the training set and evaluated on the testing set. The accuracy and confusion matrix are computed to assess the model's performance.
Support Vector Machine

The Support Vector Machine model is trained on the training set and evaluated on the testing set. The accuracy score is computed to assess the model's performance.
K-Nearest Neighbors

The K-Nearest Neighbors model is trained on the training set and evaluated on the testing set. The accuracy score is computed to assess the model's performance.
Naive Bayes

The Naive Bayes model is trained on the training set and evaluated on the testing set. The accuracy score is computed to assess the model's performance.
## Results

    - Logistic Regression: Accuracy: 97.78%
    - Support Vector Machine: Accuracy: 95.56%
    - K-Nearest Neighbors: Accuracy: 95.56%
    - Naive Bayes: Accuracy: 95.56%

## Conclusion

This project demonstrates that the Iris dataset can be effectively classified using several machine learning algorithms with high accuracy. Logistic Regression performed slightly better than the other models in this analysis. However, Support Vector Machine, K-Nearest Neighbors, and Naive Bayes also showed strong performance, indicating their suitability for this classification task. This project serves as a practical application of these algorithms and their evaluation on a standard dataset.
