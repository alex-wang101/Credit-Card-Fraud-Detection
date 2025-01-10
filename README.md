# Credit Card Fraud Detection #

This project trains and tests pre-built ai-models for credit card fraud detection. We created an algorithm to test the accuracy of each model and return the most effective model for potential commercial use. 

## Table of contents ## 
1. Introduction
2. Dataset analysis
3. Installation and Setup
4. Project Structure
5. Methodology
6. Results
7. Technologies Used
8. Contributing
9. Acknowledgments

## Introduction ##
Ever since the invention of credit cards in 1950, there has been a constant increase in credit card fraud throughout the years, reaching a total loss of $32 billion as of 2021 around the world. In order to combat this global issue controversy, we decided to create a project that tests and outputs the most effective AI model for a given randomized data set. 

### Objectives ###
Build and evaluate machine learning models to detect fraudulent transactions.
Minimize false positives while ensuring high detection accuracy.
Provide insights into transaction patterns using exploratory data analysis (EDA).
## Dataset analysis ##
The dataset used for this project is the [Credit Card Fraud Detection Dataset.](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**Total record**: 
+ 284806 transactions each with 32 attributes 
+ 492 Fraud transactions
+ 284315 not Fraud
**Attributes**: 
+ 28 PCA-transformed data that influences the fraud probability
+ Time (seconds elapsed between the specific transaction vs the first)
+ Transaction amount
+ Class (either 0 or 1) - determines whether the transaction is fraud or not fraud. 


## The Method ## 
+ How did we set up our data for training?
    Exploratory Data Analysis: 
        Analyzed transaction amounts, time distributions, and correlations between features.
        Visualized fraud vs. legitimate transaction patterns.
  
+ What are some ways we manipulated the preprocessing data?
  Scaled features using StandardScaler.
  Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique).

+ What ai models did we train and use?
  Logistic Regression
  Random Forest
  Gradient Boosting (e.g., XGBoost, LightGBM)

+ How did we evaluate the data?
  Precision: has a section for fraud and not fraud - determines the accuracy of
  Recall
  F1-Score
  Overall accuracy
4. Hyperparameter Tuning:
Used GridSearchCV to optimize model performance.
