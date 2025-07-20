# Heart Disease Prediction using Machine Learning 💖
Heart Disease Prediction

Description 📝
This project aims to predict the likelihood of heart disease using various machine learning techniques. Its primary goal is to classify whether an individual has heart disease or not, based on a set of medical attributes. This is a classification problem, where the model predicts 'has heart disease' or 'does not have heart disease'.

Features ✨
Data Loading: Loads the heart disease dataset from a CSV file.

Data Preprocessing: Handles missing values and prepares the data for the machine learning model.

Feature Scaling: Standardizes numerical features to ensure optimal model performance.

Data Splitting: Divides the dataset into training and testing sets for robust model evaluation.

Model Training: Utilizes a Logistic Regression model to predict the presence of heart disease.

Model Evaluation: Assesses the model's performance using key metrics like Accuracy, Classification Report (Precision, Recall, F1-score), and Confusion Matrix.

Dataset 📊
This project uses the Cleveland Heart Disease Dataset.

File Name: heart_disease_cleveland.csv

Source: UCI Machine Learning Repository

Description: This dataset contains 303 records and 14 attributes, including patient age, sex, chest pain type, resting blood pressure, cholesterol levels, and the presence of heart disease (the target variable).

Technologies Used 🛠️
Python

Libraries:

pandas (for data manipulation)

numpy (for numerical operations)

scikit-learn (for machine learning models and utilities)

matplotlib (for plotting and visualizations)

seaborn (for enhanced data visualizations)

Setup Instructions 🚀
Follow these steps to set up and run the project on your local machine:

Clone the Repository:

git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName/ml_task_4_folder # Navigate to your project's main folder

(Replace YourUsername and YourRepoName with your actual GitHub username and repository name. Also, replace ml_task_4_folder with the specific folder path where your heart_disease_prediction.py file resides, e.g., code alpha/ml task 2/ml task 4)

Create a Virtual Environment:
It's best practice to create a virtual environment to manage project dependencies effectively.

python -m venv ml_venv

Activate the Virtual Environment:

Windows:

.\ml_venv\Scripts\activate

macOS/Linux:

source ml_venv/bin/activate

(Your terminal prompt should now start with (ml_venv).)

Install Dependencies:
Once the virtual environment is active, install all required libraries using the requirements.txt file.

pip install -r requirements.txt

(If you don't have a requirements.txt file, you can create one by running pip freeze > requirements.txt after activating your ml_venv, and then install them manually: pip install pandas numpy scikit-learn matplotlib seaborn)

Place the Dataset:
Ensure that the heart_disease_cleveland.csv file is placed in the same folder as your heart_disease_prediction.py script.

Usage ▶️
To run the project, execute the main script while your virtual environment is activated:

python heart_disease_prediction.py

This script will process the data, train the model, and print the evaluation results to your terminal.

Results and Evaluation 📈
The model's performance was evaluated using the following metrics:

Accuracy: The overall percentage of correctly predicted instances by the model.

Precision: Out of all cases predicted as positive, how many were actually positive.

Recall: Out of all actual positive cases, how many were correctly identified by the model.

F1-Score: The harmonic mean of Precision and Recall, providing a balance between the two.

Confusion Matrix: A detailed breakdown showing true positives, true negatives, false positives, and false negatives.

Sample Output (Values may vary slightly based on your run):

--- Model Evaluation ---
Accuracy: 0.8876

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.90      0.90        30
           1       0.88      0.87      0.88        29

    accuracy                           0.89        59
   macro avg       0.89      0.89      0.89        59
weighted avg       0.89      0.89      0.89        59

Confusion Matrix:
[[27  3]
 [ 4 25]]

(You can update the Accuracy, Classification Report, and Confusion Matrix values here with your actual output from running the script.)
