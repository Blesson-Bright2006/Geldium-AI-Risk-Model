import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. LOAD THE DATA (Using the path you just generated!)
# We add 'credit_risk_dataset.csv' to the end of your path
data_path = r"C:\Users\HP\.cache\kagglehub\datasets\laotse\credit-risk-dataset\versions\1\credit_risk_dataset.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded {len(df)} rows of data, bruhh!")

# 2. QUICK CLEANING (Big data usually has empty spots)
df = df.dropna() 

# 3. PICKING FEATURES
# We'll use Income, Age, and Loan Amount to predict if they default
X = df[['person_income', 'person_age', 'loan_amnt']]
y = df['loan_status']

# 4. SPLIT AND TRAIN (80% for learning, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training the High-Accuracy XGBoost Model, vro...")
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 5. THE FINAL SCORE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🎯 AI Accuracy: {accuracy * 100:.2f}%")