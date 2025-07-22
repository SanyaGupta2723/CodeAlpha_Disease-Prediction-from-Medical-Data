# 🩺 Disease Prediction from Medical Data 🧬

This project uses machine learning models to predict the **presence of diseases** (e.g., Heart Disease) based on patient data such as age, symptoms, blood test results, and more. It covers a complete ML pipeline from preprocessing to prediction.

## 📂 Project Structure

├── disease_prediction.ipynb # Main notebook with data loading, training, and evaluation 

├── heart_disease.csv # Dataset (e.g., UCI Cleveland dataset)

├── model.pkl # Trained model file (saved via joblib)

├── scaler.pkl # Saved StandardScaler object

├── README.md # Project documentation


## 📦 Dependencies

Make sure to install the following Python libraries before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
```

🗂️ Dataset Used

Name: Heart Disease Cleveland Dataset
Source: UCI Machine Learning Repository
Attributes include: age, sex, cp (chest pain), trestbps, chol, thal, ca, etc.
Target Column: target
0 → No disease
1 → Presence of disease (converted from original multi-class 0–4 to binary)

🔄 Workflow Summary

🔹 1. Data Loading
```
df = pd.read_csv("heart_disease.csv")
```
🔹 2. Data Cleaning & Preprocessing
Remove rows with ? in ca or thal
Convert target to binary (0: no disease, 1: disease)
Convert object columns to int
Feature scaling using StandardScaler

```from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

🔹 3. Train-Test Split
```from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
```

🧠 Model Training
We used Logistic Regression as the primary model, along with other models for comparison:
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

📊 Model Evaluation
Evaluated with Accuracy, Confusion Matrix, and Classification Report.
```
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

💾 Model Saving & Inference
```
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# For loading later
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
```

✅ Output
The trained model achieved:
High Accuracy
Strong Recall (important to minimize false negatives in health)
Interpretable results with a clear confusion matrix


🔍 Algorithms Used
Logistic Regression ✅
Random Forest 🌲
XGBoost ⚡
SVM 🧠

🔬 Datasets Explored
✅ Heart Disease Dataset

(Optional Future Scope: Diabetes, Breast Cancer – also from UCI)


🚀 Future Enhancements
Use SHAP or LIME for explainability
Deploy using Flask / Streamlit
Add user-input interface for real-time prediction






