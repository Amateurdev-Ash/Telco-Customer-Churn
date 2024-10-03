# Customer Churn Prediction

This project predicts customer churn in the telecommunications industry using machine learning and deep learning. It utilizes the Telco Customer Churn dataset from Kaggle, which includes customer attributes, service details, and churn status. The goal is to identify customers at risk of churning.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
    - [Logistic Regression](#logistic-regression)
    - [Random Forest](#random-forest)
    - [Simple Neural Network](#simple-neural-network)
    - [Deep Neural Network](#deep-neural-network)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)

## Project Description
Customer churn is a major concern for telecom companies. This project aims to predict churn using machine learning and deep learning models. By identifying potential churners, companies can implement proactive retention strategies.

## Dataset
The Telco Customer Churn dataset from Kaggle is used. It contains information about customers, including:

- **Demographics:** Gender, age (SeniorCitizen), marital status (Partner), dependents (Dependents).
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
- **Contract:**   
 Contract term, PaperlessBilling, PaymentMethod.
- **Billing:** MonthlyCharges, TotalCharges.
- **Churn:** Whether the customer churned (Yes/No).

## Data Preprocessing
- **Handling Missing Values:**  Missing values in the `TotalCharges` column are identified and removed.
- **Encoding Categorical Features:**
    - **Label Encoding:** Binary features (Yes/No) are label encoded (0/1).
    - **One-Hot Encoding:**  Categorical features with multiple categories are one-hot encoded.
- **Scaling Numerical Features:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardized using `StandardScaler`.

## Exploratory Data Analysis
- **Churn Distribution:** A pie chart visualizes the proportion of churned and non-churned customers.
- **Churn vs. Monthly Charges:** A violin plot explores the relationship between monthly charges and churn.
- **Churn vs. Tenure:** A violin plot examines the relationship between customer tenure and churn.

## Modeling
### Logistic Regression
A logistic regression model is trained as a baseline classifier.

### Random Forest
A random forest classifier is employed to potentially improve prediction accuracy.

### Simple Neural Network
A simple neural network with one hidden layer is implemented using TensorFlow/Keras.

### Deep Neural Network
A deeper neural network with three hidden layers, dropout, and L2 regularization is implemented. Checkpointing and early stopping are used to optimize training.

## Evaluation
Models are evaluated using:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**

## How to Run
1. **Clone the repository:** `git clone https://github.com/your-username/churn-prediction.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the notebook:** `jupyter notebook churn_analysis.ipynb`
