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
2. `train_np[:, :-1]`: Selects all rows (:) and all columns except the last one (:-1). These are the feature columns.
<br>
3. `train_np[:, -1]`: Selects all rows and only the last column (-1). This is the target label column.
<br>
4. Dataset broken into Training data, testing data and validation data. Training and validation trains and refines the model, whereas Testing is hidden except for RandomForest.
  
5. What are some ways we manipulated the preprocessing data?
    - Scaled features using StandardScaler.
    - Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique).
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
4
