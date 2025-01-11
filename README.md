<img width="816" alt="image" src="https://github.com/user-attachments/assets/62ab8fab-8bf8-4b9c-a6ee-196ad122aebe" />


**We tested 4 pre-built AI models to see which one is more effective for the given data.**
<br>
![GitHub contributors](https://img.shields.io/github/contributors/alex-wang101/Credit-Card-Fraud-Detection)



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


## Setting Up ## 
1. We analyzed transaction amounts, time distributions, and correlations between features
2. Visualized fraud vs. legitimate transaction patterns
3. Grouped the data into Pandas dataframe

+ How did we set the data for the models to read? 
<img width="679" alt="image" src="https://github.com/user-attachments/assets/f1fd4cf0-8dae-4fcf-9d4f-834bb84243e6" />
<br>
1. Converts the dataset in the Pandas dataframe into NumPy arrays into features `x` and target labels `y`
       <br>
2. **train_np[:, :-1]**: Selects all rows (:) and all columns except the last one (:-1). These are the feature columns.
<br>
3. **train_np[:, -1]**: Selects all rows and only the last column (-1). This is the target label column.
<br>
4. Dataset broken into Training data, testing data and validation data. Training and validation trains and refines the model, whereas Testing is hidden except for RandomForest.  
5. What are some ways we manipulated the preprocessing data?
    - Scaled features using StandardScaler.
    - Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique), where we randomly took out not fraud data so the number of fraud transactions are equivalent to the number of not fraud transactions. 
<br>
While using the scarmbled data of every single transaction, the model often ignore the `recall negative` values 

How did we evaluate the accuracy of the model?
<br>
<img width="314" alt="image" src="https://github.com/user-attachments/assets/1c7e7af7-6760-408a-9a1e-905c4d819a3f" />
<br>
    - Precision represents "Predicted Fraud"
  <br>
    - Recall represents "Predicted Not Fraud"
  <br>
    - "True/False" represents if the transaction is evaluated correctly, and "+/-" represents if the transaction is Fraud or Not Fraud
Example of our model reports:
<br>
<img width="199" alt="image" src="https://github.com/user-attachments/assets/415971c4-e96d-439a-a6f7-f5843e7017ef" />
<br>
    - Examplar report data from one of our testings; each fraction represents the percent of success in that catagory.
<br>

## AI Models Used in the Credit Card Fraud Detection Project

Below is a detailed description of the AI models used in this project:

### 1. Logistic Regression
![image](https://github.com/user-attachments/assets/d0354c9a-7660-42f8-8be8-6e6f2c3e5457)
<br>

- **Description**: Logistic Regression is a statistical model that predicts the probability of a binary outcome. It models the relationship between the dependent variable and one or more independent variables using a logistic function.
- **Application in Fraud Detection**: It estimates the likelihood of a transaction being fraudulent based on input features. Despite its simplicity, it serves as a strong baseline for binary classification tasks.
- **Advantages**:
  - Interpretable coefficients indicating feature importance.
  - Efficient training on large datasets.
- **Considerations**:
  - Assumes a linear relationship between features and the log-odds of the outcome.
  - May underperform with complex, non-linear data patterns.

### 2. Decision Tree Classifier
![image](https://github.com/user-attachments/assets/f0a12182-d328-47e3-987c-ae7d6ab90fb7)
<br>
- **Description**: A Decision Tree is a flowchart-like structure where internal nodes represent feature tests, branches represent outcomes, and leaf nodes represent class labels.
- **Application in Fraud Detection**: It segments the dataset based on feature values to classify transactions as fraudulent or non-fraudulent.
- **Advantages**:
  - Easy to interpret and visualize.
  - Captures non-linear relationships between features and the target variable.
- **Considerations**:
  - Prone to overfitting, especially with deep trees.
  - Sensitive to small variations in the data.

### 3. Random Forest Classifier
![image](https://github.com/user-attachments/assets/c20513cb-e0be-4e24-868e-b052d425990b)
<br>
- **Description**: Random Forest is an ensemble method that constructs multiple decision trees during training and outputs the mode of their predictions.
- **Application in Fraud Detection**: It aggregates the decisions of various trees to improve predictive accuracy and control overfitting.
- **Advantages**:
  - Reduces overfitting compared to individual decision trees.
  - Handles large datasets with higher dimensionality.
- **Considerations**:
  - Less interpretable than single decision trees.
  - Requires more computational resources for training and prediction.

### 4. Gradient Boosting Classifier
![image](https://github.com/user-attachments/assets/ffbc073f-8cd7-4710-882c-322cedd0323d)
<br>
- **Description**: Gradient Boosting builds an ensemble of trees sequentially, where each tree corrects the errors of its predecessors by optimizing a loss function.
- **Application in Fraud Detection**: It iteratively improves model performance by focusing on misclassified transactions, enhancing detection accuracy.
- **Advantages**:
  - High predictive performance, especially with complex datasets.
  - Effectively handles imbalanced data by focusing on difficult cases.
- **Considerations**:
  - Can overfit if not properly tuned.
  - Training can be time-consuming, especially with large datasets.



---

Each of these models has unique strengths in detecting fraudulent transactions. The choice of the model depends on the dataset size, feature complexity, interpretability requirements, and computational resources. In this project, multiple models were evaluated to determine the most effective approach.

## Summary and Conclusion

This project aimed to develop a machine learning-based solution to detect fraudulent transactions effectively. Several machine learning models, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Artificial Neural Networks (ANN), were evaluated for their performance.

## Summary and Conclusion

This project evaluated several machine learning models for credit card fraud detection. Below are the key findings with performance metrics:

### Model Performance Metrics:

1. **Linear Model (Logistic Regression)**:
   - **Precision (Fraud)**: 0.77  
   - **Recall (Fraud)**: 0.65  
   - **F1-Score (Fraud)**: 0.71  
   - **Accuracy**: 1.00  

2. **Neural Predictions (Unbalanced)**:
   - **Precision (Fraud)**: 0.95  
   - **Recall (Fraud)**: 0.87  
   - **F1-Score (Fraud)**: 0.91  
   - **Accuracy**: 1.00  

3. **Random Forest Classifier**:
   - **Precision (Fraud)**: 1.00  
   - **Recall (Fraud)**: 0.80  
   - **F1-Score (Fraud)**: 0.89  
   - **Accuracy**: 1.00  

4. **Gradient Boosting Classifier (Unbalanced)**:
   - **Precision (Fraud)**: 0.90  
   - **Recall (Fraud)**: 0.76  
   - **F1-Score (Fraud)**: 0.82  
   - **Accuracy**: 1.00  

5. **Balanced Neural Predictions**:
   - **Precision (Fraud)**: 0.84  
   - **Recall (Fraud)**: 1.00  
   - **F1-Score (Fraud)**: 0.91  
   - **Accuracy**: 0.90  

6. **Balanced Logistic Regression**:
   - **Precision (Fraud)**: 0.94  
   - **Recall (Fraud)**: 0.93  
   - **F1-Score (Fraud)**: 0.95  
   - **Accuracy**: 0.95  

7. **Balanced Random Forest Classifier (*Best Model*)**:
   - **Precision (Fraud)**: 0.95  
   - **Recall (Fraud)**: 0.99  
   - **F1-Score (Fraud)**: 0.97  
   - **Accuracy**: 0.97  

8. **Balanced Gradient Boosting Classifier**:
   - **Precision (Fraud)**: 0.92  
   - **Recall (Fraud)**: 0.96  
   - **F1-Score (Fraud)**: 0.94  
   - **Accuracy**: 0.94  

### Conclusion:
The **Balanced Random Forest Classifier** emerged as the top-performing model with a precision of 0.95, recall of 0.99, and an F1-score of 0.97. These metrics highlight its strong ability to correctly identify fraudulent transactions while minimizing false negatives and false positives. 

Other models, such as the Balanced Gradient Boosting Classifier and Balanced Logistic Regression, also showed competitive performance. The unbalanced models performed well in terms of accuracy but struggled with recall for fraud detection due to class imbalance.

This project demonstrates that balanced ensemble methods, like the Random Forest and Gradient Boosting Classifiers, are highly effective in detecting fraudulent transactions. Future improvements could involve further hyperparameter tuning, implementing real-time detection pipelines, or exploring advanced deep learning techniques.
