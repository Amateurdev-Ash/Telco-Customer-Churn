# Telco-Customer-Churn
##Customer Churn Prediction Case Study
###Project Overview
This project focuses on building a machine learning classifier to predict customer churn based on a Telco Customer Churn dataset from a Kaggle competition. The goal is to classify whether a customer is likely to churn based on their demographic details, services subscribed, and billing information. This repository provides the complete workflow from data preprocessing and exploratory data analysis to model building and evaluation.

###Table of Contents
1.	Dataset
2.	Problem Statement
3.	Project Workflow
4.	Data Preprocessing
5.	Exploratory Data Analysis (EDA)
6.	Feature Engineering
7.	Modeling
8.	Results
9.	Tools Used
10.	Conclusion

###The dataset used in this project is the Telco Customer Churn dataset, sourced from Kaggle. The dataset contains 7,043 customer records with the following features:

•	####customerID: Unique identifier for each customer

•	####Demographic details (e.g., gender, SeniorCitizen, Partner, Dependents)

•	####Service details (e.g., tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.)

•	####Billing details (e.g., MonthlyCharges, TotalCharges, PaymentMethod)

•	####Target: Churn – whether the customer churned (Yes/No)


###Problem Statement

The objective is to build a predictive model that can estimate the likelihood of a customer churning (leaving the telecom company) based on their previous behaviors and subscribed services. This will help the telecom company to identify at-risk customers and take preventive measures to reduce churn.




###Project Workflow

####Data Cleaning: Identifying and handling missing data in the TotalCharges field

####Data Preprocessing: Label encoding, one-hot encoding, and feature scaling

####Exploratory Data Analysis (EDA): Analyzing relationships between features and churn through visualizations

####Modeling: Building and evaluating machine learning models

####Evaluation: Using metrics like accuracy, precision, recall, and AUC to measure the model’s performance




###Data Preprocessing

####Steps Involved

•	####Handling Missing Values: Identified 11 missing values in the TotalCharges column. These were found to be space characters, which were removed

•	####Label Encoding: Categorical binary columns (e.g., gender, Partner, Dependents) were label-encoded (Yes = 1, No = 0)

•	####One-Hot Encoding: Non-binary categorical columns (e.g., InternetService, PaymentMethod) were converted to numerical features using one-hot encoding

•	####Feature Scaling: Numerical columns (tenure, MonthlyCharges, TotalCharges) were scaled using the StandardScaler to normalize their distribution for better performance in models

•	####Dropping Irrelevant Features: The customerID column was removed since it does not contribute to predicting churn

Exploratory Data Analysis (EDA)

•	Distribution of Churn: Created visualizations (e.g., pie charts) to show the percentage of customers who churned vs. those who did not

•	Crosstab Analysis: Performed crosstab analysis to understand churn behavior across different demographic and service-related categories, such as gender, SeniorCitizen, Partner, Dependents, and Contract.
•	Violin Plots: Used violin plots to visualize the relationship between churn and features like MonthlyCharges and tenure

Found that customers with higher monthly charges and shorter tenures tend to churn more.




Feature Engineering

After cleaning and encoding, we had 42 Features including the target (Churn).

Numerical columns were scaled, and categorical variables were encoded to make the data ready for model training.




Modeling

The prepared dataset was used to train several machine learning classifiers, including:

•	Logistic Regression

•	Decision Trees

•	Random Forest

•	Tensorflow Sequential (Convolutional Neural Network)



Model Evaluation

Metrics used for evaluating model performance:

•	Accuracy: Measures the percentage of correct predictions

•	Precision: Measures the accuracy of positive predictions (customers who churned)

•	Recall: Measures the ability to capture all positive instances (customers who churned)

•	AUC-ROC Curve: Plots the true positive rate vs. false positive rate and summarizes model performance




Results

Initial tests showed that certain models (e.g., Random Forest) performed better in terms of AUC and accuracy compared to Logistic Regression

Key insights from the data:

•	Higher Monthly Charges: Customers with higher monthly charges are more likely to churn

•	Shorter Tenure: New customers tend to churn more, suggesting dissatisfaction with the service early on

•	Senior Citizens: Senior citizens have a higher churn rate compared to younger customers

•	Contract Type: Customers with month-to-month contracts are more prone to churn compared to those with yearly contracts

Tools Used
•	Python 3.9
•	Pandas for data manipulation
•	Matplotlib and Seaborn for data visualization
•	Scikit-learn for data preprocessing, model building, and evaluation
•	Tensorflow for advanced modeling
•	Jupyter Notebooks for running and documenting experiments




Conclusion

The project successfully built a churn prediction model that can be used by telecom companies to identify customers who are at risk of leaving. By identifying the key drivers of churn (e.g., higher charges, short tenure), the company can implement strategic actions to retain customers.
