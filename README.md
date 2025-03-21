<img width="812" alt="image" src="https://github.com/user-attachments/assets/1f7fae10-5b75-4b52-9722-4fa09fa97e38" />


**We tested 5 AI models to see which one is more effective for the given data.**
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

### 2. Neural Network
![A-simple-neural-network-with-two-hidden-layers-of-two-nodes-each-four-inputs-and-a](https://github.com/user-attachments/assets/fd37aea3-78d4-4e1f-8928-3f8a8406381b)
<br>

- **Description**: A neural network utilizing dense (fully connected) layers predicts outcomes by connecting all neurons between layers, enabling the model to learn complex relationships in the data. Each dense layer processes input features through weighted connections and activation functions, making this architecture highly effective for tasks requiring intricate feature interactions.
- **Application in Fraud Detection**: Dense layers in neural networks are employed to analyze transactional data (e.g., amount, location, timestamp) and identify patterns associated with fraudulent activities. The network learns complex relationships between features, improving detection accuracy over traditional models.

- **Advantages**:
  - **Effective Handling of Complex Data**: Excels at identifying non-linear and intricate patterns in fraud-related data.
  - **Multi-Feature Integration**: Simultaneously processes multiple input features for improved prediction accuracy.
  - **Scalability**: Capable of continuous improvement with growing datasets.

- **Considerations**:
  - **Data Requirements**: Requires substantial amounts of labeled data for effective training, which may not always be readily available.
  - **Risk of Overfitting**: Needs proper regularization techniques (e.g., dropout, L2 regularization) to avoid overfitting on training data.
  - **Computational Complexity**: High computational demands, especially for real-time detection in large-scale systems.


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
### 5. Support Vectone Machine
![1701290431943](https://github.com/user-attachments/assets/a4aadb88-7038-48f0-a1d0-0e0678841a50)
<br>
- **Description**: A Support Vector Machine (SVM) is a supervised learning model that finds the optimal hyperplane to separate data points of different classes. It can use kernel functions to map data into higher dimensions for better separability in non-linear datasets.
- **Application in Fraud Detection**: It classifies transactions as fraudulent or legitimate by finding patterns in input features (e.g., transaction amount, time, location) and leveraging kernels to handle non-linear relationships.
- **Advantages**:
  - Performs well in high-dimensional spaces.
  - Effective at capturing complex, non-linear patterns using kernel functions.
  - Robust to overfitting, especially with a small number of samples relative to features.
- **Considerations**:
  - Computationally expensive, particularly with large datasets.
  - Requires careful tuning of kernel type and parameters for optimal performance.
  - Not ideal for real-time detection in large-scale systems due to resource demands.


---

Each of these models has unique strengths in detecting fraudulent transactions. The choice of the model depends on the dataset size, feature complexity, interpretability requirements, and computational resources. In this project, multiple models were evaluated to determine the most effective approach.

## Summary and Conclusion

This project aimed to develop a machine learning-based solution to detect fraudulent transactions effectively. Several machine learning models, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Artificial Neural Networks (ANN), were evaluated for their performance.

## Summary and Conclusion

This project evaluated several machine learning models for credit card fraud detection. Below are the key findings with performance metrics:

### Model Performance Metrics:

1. **Unbalanced Linear Model**:
   - **Precision (Fraud)**: 0.77  
   - **Recall (Fraud)**: 0.65  
   - **F1-Score (Fraud)**: 0.71  
   - **Accuracy**: 1.00  

2. **Unbalanced Neural Network**:
   - **Precision (Fraud)**: 0.95  
   - **Recall (Fraud)**: 0.87  
   - **F1-Score (Fraud)**: 0.91  
   - **Accuracy**: 1.00  

3. **Unbalanced Random Forest Classifier**:
   - **Precision (Fraud)**: 1.00  
   - **Recall (Fraud)**: 0.80  
   - **F1-Score (Fraud)**: 0.89  
   - **Accuracy**: 1.00  

4. **Unbalanced Gradient Boosting Classifier**:
   - **Precision (Fraud)**: 0.90  
   - **Recall (Fraud)**: 0.76  
   - **F1-Score (Fraud)**: 0.82  
   - **Accuracy**: 1.00
  
5. **Unbalanced Support Vector Machine**
   - **Precision (Fraud)**: 0.09  
   - **Recall (Fraud)**: 0.91  
   - **F1-Score (Fraud)**: 0.16  
   - **Accuracy**: 0.98
     
7. **Balanced Logistic Regression**:
   - **Precision (Fraud)**: 0.97  
   - **Recall (Fraud)**: 0.93  
   - **F1-Score (Fraud)**: 0.95  
   - **Accuracy**: 0.95  

8. **Balanced Neural Network**:
   - **Precision (Fraud)**: 1.00  
   - **Recall (Fraud)**: 0.80  
   - **F1-Score (Fraud)**: 0.89  
   - **Accuracy**: 0.90  

9. **Balanced Random Forest Classifier (*Best Model*)**:
   - **Precision (Fraud)**: 0.99  
   - **Recall (Fraud)**: 0.95  
   - **F1-Score (Fraud)**: 0.97  
   - **Accuracy**: 0.97  

10. **Balanced Gradient Boosting Classifier**:
    - **Precision (Fraud)**: 0.96  
    - **Recall (Fraud)**: 0.92  
    - **F1-Score (Fraud)**: 0.94  
    - **Accuracy**: 0.94
     
10. **Balanced Support Vector Machine**
    - **Precision (Fraud)**: 0.99  
    - **Recall (Fraud)**: 0.93  
    - **F1-Score (Fraud)**: 0.96  
    - **Accuracy**: 0.96

### Conclusion:
The **Balanced Random Forest Classifier** emerged as the top-performing model with a precision of 0.99, recall of 0.95, and an F1-score of 0.97. These metrics highlight its strong ability to correctly identify fraudulent transactions while minimizing false negatives and false positives. 

Other models, such as the Balanced Gradient Boosting Classifier and Balanced Logistic Regression, also showed competitive performance. The unbalanced models performed well in terms of accuracy but struggled with recall for fraud detection due to class imbalance.

This project demonstrates that balanced ensemble methods, like the Random Forest and Gradient Boosting Classifiers, are highly effective in detecting fraudulent transactions. Future improvements could involve further hyperparameter tuning, implementing real-time detection pipelines, or exploring advanced deep learning techniques.
