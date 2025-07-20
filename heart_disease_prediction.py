# heart_disease_prediction.py
#python -m venv ml_venv
#.\ml_venv\Scripts\activate

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Added for Logistic Regression model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # For better visualizations

# --- Configuration ---
# IMPORTANT: Set the correct path to your heart_disease_cleveland.csv file
# Since the CSV file is in the same folder as this script, use a relative path.
DATA_PATH = "heart_disease_cleveland.csv" # <-- This should be correct now

# Define column names as the dataset does not come with a header
# These names are standard for the Cleveland Heart Disease dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# --- Step 1: Load the Dataset ---
print("Loading the Heart Disease dataset...")
try:
    # Read the CSV file. '?' is treated as a missing value.
    # header=None because the file does not have column names in the first row.
    df = pd.read_csv(DATA_PATH, names=column_names, na_values='?')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {DATA_PATH}")
    print("Please check the DATA_PATH in the script and ensure the CSV file is in the correct folder.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Step 2: Initial Data Inspection ---
print("\n--- Dataset Head (first 5 rows) ---")
print(df.head()) # First 5 rows of the DataFrame

print("\n--- Dataset Info (column types and non-null counts) ---")
df.info() # Information about the DataFrame, including data types and missing values

print("\n--- Dataset Description (statistical summary) ---")
print(df.describe()) # Statistical summary of numerical columns

print("\n--- Missing Values Count ---")
print(df.isnull().sum()) # Count of missing values per column

# --- Step 3: Handle Missing Values ---
# The 'ca' and 'thal' columns often have missing values ('?') which pandas reads as NaN.
# For simplicity, we will drop rows with any missing values for now.
# In a real project, you might impute (fill) these values using more advanced techniques.
print("\n--- Handling Missing Values ---")
initial_rows = df.shape[0]
df.dropna(inplace=True) # Drop rows with any NaN values
rows_after_dropping = df.shape[0]
print(f"Initial rows: {initial_rows}")
print(f"Rows after dropping missing values: {rows_after_dropping}")
print(f"Dropped {initial_rows - rows_after_dropping} rows with missing values.")

print("\n--- Missing Values Count After Dropping ---")
print(df.isnull().sum()) # Verify no more missing values

# --- Step 4: Convert 'target' column to binary (0 or 1) ---
# The 'target' column has values 0, 1, 2, 3, 4.
# 0 means no heart disease, 1,2,3,4 mean presence of heart disease.
# We will convert it to a binary classification problem: 0 (no disease) or 1 (disease).
print("\n--- Converting 'target' to Binary ---")
print(f"Original unique values in 'target': {df['target'].unique()}")
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
print(f"New unique values in 'target': {df['target'].unique()}")
print("Target column converted to binary (0: No Disease, 1: Disease).")

# --- Step 5: Convert 'ca' and 'thal' to numeric (they might still be objects if not handled correctly) ---
# Even after dropping NaNs, sometimes these columns might remain 'object' type if not all '?' were removed
# or if other non-numeric characters exist. Let's ensure they are numeric.
# Errors='coerce' will turn non-numeric values into NaN, which we already dropped.
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

# After converting, drop any new NaNs that might have been introduced by 'coerce' (though unlikely after initial dropna)
df.dropna(inplace=True)

print("\n--- Data Types After Conversions ---")
df.info()

print("\nInitial data loading and preprocessing complete. Ready for feature engineering and model training.")

# --- Step 6: Separate Features (X) and Target (y) ---
# The 'target' column is what we want to predict. All other columns are features.
print("\n--- Separating Features (X) and Target (y) ---")
X = df.drop('target', axis=1) # Features are all columns except 'target'
y = df['target'] # Target is the 'target' column

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- Step 7: Split Data into Training and Testing Sets ---
# We split the data to train the model on one part and test its performance on unseen data.
# test_size=0.2 means 20% of data will be used for testing, 80% for training.
# random_state ensures reproducibility of the split.
# stratify=y ensures that the proportion of target classes is the same in both train and test sets.
print("\n--- Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- Step 8: Feature Scaling (Standardization) ---
# Scaling is crucial for many ML algorithms (like SVM, Logistic Regression)
# It transforms features to have a mean of 0 and a standard deviation of 1.
# We fit the scaler only on the training data to prevent data leakage.
print("\n--- Applying Feature Scaling (Standardization) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform on training data
X_test_scaled = scaler.transform(X_test) # Transform only on test data using the fitted scaler

print("Features scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# --- Step 9: Model Training (Example: Logistic Regression) ---
# We'll start with Logistic Regression, a simple yet effective classification algorithm.
print("\n--- Training Logistic Regression Model ---")
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
model.fit(X_train_scaled, y_train)
print("Logistic Regression Model trained successfully.")

# --- Step 10: Model Evaluation ---
print("\n--- Evaluating Logistic Regression Model ---")
y_pred = model.predict(X_test_scaled)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate Classification Report
# This shows precision, recall, f1-score for each class (0: No Disease, 1: Disease)
report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
print("\nClassification Report:")
print(report)

# Generate Confusion Matrix
# Helps visualize the performance of the classification model.
# Rows are actual classes, columns are predicted classes.
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Optional: Plotting Confusion Matrix for better visualization
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

print("\nHeart Disease Prediction Task 4 (Logistic Regression) complete.")
