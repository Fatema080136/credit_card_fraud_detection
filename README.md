# credit_card_fraud_detection

# Dataset [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]

The dataset is available on Kaggle and focuses on credit card transactions. It contains 284,807 data points with 31 features each. 28 of these features are numerical representations obtained through Principal Component Analysis (PCA) to protect customer privacy. The remaining three features are:
* Time: This represents the elapsed seconds between a transaction and the first transaction in the dataset.
* Amount: This indicates the transaction amount.
* Class: This is a binary variable (0 or 1) where 1 signifies a fraudulent transaction and 0 represents a legitimate transaction.

# Classifiers
  1. XGBoost
  2. Random forest

# Data oversampling method
  1. RandomOverSampler
  2. SMOTE

# Feature selection methods:
  1. PCA
  2. Forward selection method
     
# Confusion matrix
![CM](https://github.com/user-attachments/assets/36b8cf49-bf0c-4757-90a7-d3e3d5c3afdc)

# Precision and recall
![PR_fraud_detection](https://github.com/user-attachments/assets/1894c788-c8d8-48b3-b42c-6289f9d9bfcb)
