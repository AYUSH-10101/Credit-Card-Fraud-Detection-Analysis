# Credit Card Fraud Detection Analysis

This project focuses on analyzing credit card transaction data to identify fraudulent activities. It utilizes machine learning techniques, specifically Logistic Regression, to build a predictive model. The analysis includes data loading, preprocessing, handling imbalanced data through undersampling, model training, and evaluation.

## Dataset

The dataset used in this analysis is the "Credit Card Fraud Detection" dataset available on Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, as the positive class (frauds) account for only 0.172% of all transactions.

Due to confidentiality, the original features are not provided and are replaced with principal components obtained through PCA transformation, except for 'Time' and 'Amount'. The 'Class' column is the response variable and takes value 1 in case of fraud and 0 otherwise.

## Technologies Used

* **Python:** The primary programming language used for the analysis.
* **NumPy:** For numerical computations and array manipulation.
* **Pandas:** For data manipulation and analysis using DataFrames.
* **Scikit-learn (sklearn):** A comprehensive library for machine learning tasks, including:
    * `train_test_split`: For splitting the dataset into training and testing sets.
    * `LogisticRegression`: The classification model used for fraud detection.
    * `accuracy_score`: For evaluating the performance of the model.

## Setup and Execution

1.  **Import Dependencies:** The necessary libraries (NumPy, Pandas, Scikit-learn) are imported at the beginning of the script.

2.  **Load Dataset:** The credit card transaction data is loaded into a Pandas DataFrame using `pd.read_csv()`.

3.  **Explore Data:**
    * The first and last few rows of the dataset are displayed using `head()` and `tail()` to get an initial understanding of the data.
    * `info()` provides a summary of the dataset, including data types and non-null values.
    * `isnull().sum()` checks for any missing values in the dataset.
    * `value_counts()` on the 'Class' column reveals the significant imbalance between legitimate and fraudulent transactions.

4.  **Handle Imbalanced Data (Undersampling):**
    * The dataset is highly imbalanced, which can lead to a biased model. To address this, undersampling is performed:
        * The legitimate and fraudulent transactions are separated into different DataFrames.
        * A random sample of an equal number of legitimate transactions as fraudulent transactions is created using `legit.sample(n=492)`.
        * A new, balanced dataset (`new_dataset`) is created by concatenating the undersampled legitimate transactions and all the fraudulent transactions.
        * The `value_counts()` on the 'Class' of the `new_dataset` confirms that the classes are now balanced.

5.  **Split Data into Features and Target:**
    * The 'Class' column is separated as the target variable (Y), and the remaining columns are considered as features (X).

6.  **Split Data into Training and Testing Sets:**
    * The balanced dataset is split into training and testing sets using `train_test_split()` with an 80-20 ratio (`test_size=0.2`).
    * `stratify=Y` ensures that the proportion of fraudulent and legitimate transactions is maintained in both the training and testing sets.
    * `random_state=2` ensures reproducibility of the split.

7.  **Model Training (Logistic Regression):**
    * A Logistic Regression model is initialized using `LogisticRegression()`.
    * The model is trained using the training features (X\_train) and the training target (Y\_train) using the `fit()` method.

8.  **Model Evaluation:**
    * **Accuracy Score:** The performance of the trained model is evaluated using the accuracy score on both the training and testing datasets.
        * Predictions are made on the training data using `model.predict(X_train)`, and the accuracy is calculated by comparing these predictions with the actual training labels (Y\_train) using `accuracy_score()`.
        * Similarly, predictions are made on the testing data, and the accuracy is calculated against the actual testing labels (Y\_test).

## Results

The accuracy scores obtained from the Logistic Regression model are:

* **Accuracy on Training data:** Approximately 94.16%
* **Accuracy score on Test Data:** Approximately 93.91%

These results indicate that the Logistic Regression model, trained on the undersampled and balanced dataset, performs well in distinguishing between legitimate and fraudulent credit card transactions on unseen data.

## Description

This project provides a basic implementation of credit card fraud detection using a Logistic Regression model. It highlights the importance of addressing imbalanced datasets, a common challenge in fraud detection. The undersampling technique helps in creating a more balanced training set, leading to a more robust model. The evaluation on the test set provides an estimate of the model's performance on new, unseen data. Further improvements could involve exploring other machine learning algorithms, employing more sophisticated techniques for handling imbalanced data (like oversampling or using SMOTE), and performing feature engineering to potentially enhance the model's predictive capabilities.
