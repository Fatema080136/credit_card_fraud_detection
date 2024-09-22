#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
import sys
import os
data_path = os.path.abspath(r'Downloads\creditcard\creditcard.csv')
data = pd.read_csv(data_path)


# In[19]:


data.shape


# In[17]:


# Data overview
print(data.head())


# In[18]:


# Check for missing values
data.isnull().sum()


# In[3]:


# Standardize the 'Amount' and 'Time' features
#scaler = StandardScaler()
#data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
#data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))


# In[3]:


# Features and target
X = data.drop(columns=['Class'])  # Features
y = data['Class']  # Target (1: Fraud, 0: No Fraud)


# In[24]:


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[21]:


# Scale the features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


# In[8]:


#Balance the classes using SMOTE (oversampling minority class)
smote = SMOTE(sampling_strategy='minority', random_state=42)
#X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


# In[25]:


#Or balance the classes using RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Apply Random Oversampling to the training data
#X_train_balanced, y_train_balanced  = ros.fit_resample(X_train_scaled, y_train)

# Apply Random Oversampling to the training data
X_train_balanced, y_train_balanced  = ros.fit_resample(X_train, y_train)

# Check the new class distribution after oversampling
print("Original training class distribution:", y_train.value_counts())
print("Resampled training class distribution:",y_train_balanced.value_counts())


# In[7]:


def model_prediction_result(model, X_test):
  # Make predictions
  y_pred = model.predict(X_test)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy * 100:.2f}%")

  # Classification report
  print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

  cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = [0, 1])

  cm_display.plot()
  plt.show()


# In[24]:


from IPython.display import Image
PATH = "/Users/Fatema Tuj Johora/Downloads/"
Image(filename = PATH + "CM.png", width=400, height=300)


# In[25]:


Image(filename = PATH + "PR_fraud_detection.png", width=400, height=300)


# In[27]:


# Initialize the XGBoost classifier
model = XGBClassifier(scale_pos_weight=1,
                          eval_metric='logloss')

# Train the model
model.fit(X_train_balanced, y_train_balanced)

# Call prediction method
#model_prediction_result(model, X_test_scaled)
model_prediction_result(model, X_test)


# In[9]:


import matplotlib.pyplot as plt
plot_importance(model, importance_type='weight')
plt.show()


# In[11]:


from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[15]:


# K-fold Cross-Validation
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_balanced, y_balanced  = ros.fit_resample(X_train, y_train)
    model.fit(X_balanced, y_balanced)

    # Call prediction method
    model_prediction_result(model, X_test)


# In[10]:


# Now try with fewer features
# Step 6: Apply PCA to reduce dimensions, even though the all feature are already principal components
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_balanced)
#X_test_pca = pca.transform(X_test_scaled)
X_test_pca = pca.transform(X_test)

# Check how many principal components were retained
print(f"Number of components retained: {pca.n_components_}")


# In[11]:


# Step 7: Train an XGBoost model on the PCA-reduced data
model_pca = XGBClassifier(scale_pos_weight=1, eval_metric='logloss',
                          random_state=42)

# Train the model
model_pca.fit(X_train_pca, y_train_balanced)


# In[12]:


# Call prediction method
model_prediction_result(model_pca, X_test_pca)


# In[47]:


import matplotlib.pyplot as plt
plot_importance(model_pca, importance_type='weight')
plt.show()


# In[13]:


# Apply Forward Feature Selection (SelectKBest) instead of PCA
k_best_selector = SelectKBest(score_func=f_classif, k=19)  # Select top 19 features
X_train_kbest = k_best_selector.fit_transform(X_train_balanced, y_train_balanced)
#X_test_kbest = k_best_selector.transform(X_test_scaled)
X_test_kbest = k_best_selector.transform(X_test)

# Get the selected feature indices and corresponding scores
selected_features = k_best_selector.get_support(indices=True)
print(f"Selected feature indices: {selected_features}")
print(f"Feature scores: {k_best_selector.scores_}")


# In[14]:


# Train an XGBoost model on the selected features
model_kbest = XGBClassifier(scale_pos_weight=1, eval_metric='logloss')

model_kbest.fit(X_train_kbest, y_train_balanced)


# In[15]:


# Call prediction method
model_prediction_result(model_kbest, X_test_kbest)


# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Step 8: Train the model
#rf_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)


# In[18]:


#model_prediction_result(rf_model, X_test_scaled)
model_prediction_result(rf_model, X_test)

