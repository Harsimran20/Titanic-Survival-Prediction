#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Import data visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Import data manipulation library
import pandas as pd
# Import machine learning libraries for model building and evaluation
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.tree import DecisionTreeClassifier  # Decision tree algorithm
from sklearn.metrics import accuracy_score  # For model performance evaluation


# In[43]:


# Load the Titanic dataset from seaborn
titanic = sns.load_dataset("titanic")

# Define feature columns and target column for the model
features = ["pclass", "sex", "fare", "embarked", "age"]
target = ["survived"]

# Handle missing values in the 'age' column using median imputation
imp_median = SimpleImputer(strategy = "median")
titanic[["age"]] = imp_median.fit_transform(titanic[["age"]])

# Handle missing values in the 'embarked' column using most frequent value imputation
imp_freq = SimpleImputer(strategy = "most_frequent")
titanic[["embarked"]] = imp_freq.fit_transform(titanic[["embarked"]])

# Initialize label encoder for converting categorical variables to numerical
le = LabelEncoder()

# Encode the 'sex' column (convert categorical to numerical values)
titanic["sex"] = le.fit_transform(titanic["sex"])

# Encode the 'embarked' column (convert categorical to numerical values)
titanic["embarked"] = le.fit_transform(titanic["embarked"])

# Prepare feature matrix (X) and target vector (y)
X = titanic[features]
y = titanic["survived"]

# Split the data into training and testing sets (70% train, 30% test)
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)


# In[55]:


# Create a Decision Tree Classifier instance
model = DecisionTreeClassifier(max_depth = 4)

# Train the model using the training data
model.fit(X_train,y_train)

# Make predictions on the test set
y_pred_test = model.predict(X_test)

# Make predictions on the training set
y_pred_train = model.predict(X_train)

# Calculate and display the training accuracy
print("Training Accuracy:", accuracy_score(y_train,y_pred_train)*100,"%")

# Calculate and display the testing accuracy
print("Testing Accuracy:", accuracy_score(y_test,y_pred_test)*100, "%")


# In[57]:


from sklearn.tree import plot_tree
# Create a large figure to accommodate the decision tree visualization
plt.figure(figsize=(18,10))
# Plot the decision tree with detailed configuration
plot_tree(
    model,  # The trained decision tree model
    feature_names = X.columns,  # Use column names from features as node labels
    class_names = ["Died","Survived"],  # Labels for the target classes
    filled = True  # Fill nodes with colors based on majority class
)
# Adjust layout to prevent overlapping elements
plt.tight_layout()


# In[63]:


# Import RandomForestClassifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest classifier with 201 trees and out-of-bag scoring enabled
rf = RandomForestClassifier(
    n_estimators = 201,  # Number of decision trees in the forest
    oob_score = True,          # Enable out-of-bag score calculation
    max_depth = 4
)

# Train the Random Forest model on the training data
rf.fit(X_train,y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Display the out-of-bag score (internal validation metric)
print("OOB Score:", (rf.oob_score_)*100, "%")

# Calculate and display the testing accuracy as a percentage
print("Testing Accuracy:", accuracy_score(y_test,y_pred_test)*100, "%")


# In[ ]:




