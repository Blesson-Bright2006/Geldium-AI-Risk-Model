from flask import Flask, request, jsonify, render_template
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# --- 🧠 THE AI SYSTEM SETUP ---
# Path to your downloaded dataset
data_path = r"C:\Users\HP\.cache\kagglehub\datasets\laotse\credit-risk-dataset\versions\1\credit_risk_dataset.csv"
df = pd.read_csv(data_path).dropna()

# COMPLEX MATH: Feature Engineering
df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']
df['age_history_factor'] = df['person_age'] * df['cb_person_cred_hist_length']

# Define the "Smart" Features
features = ['person_income', 'person_age', 'loan_amnt', 'loan_to_income_ratio', 'age_history_factor']
X = df[features]
y = df['loan_status']

# Train the "Chef"
model = xgb.XGBClassifier()
model.fit(X, y)
print("✅ Geldium AI Server is TRAINED and ONLINE, bruhh!")

# --- 🌐 THE WEB API ENDPOINTS ---

# 1. The Home Route (This shows your website)
@app.route('/')
def home():
    return render_template('index.html')

# 2. The Predict Route (This does the AI math)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    
    # Calculate complex math for the new user on the fly
    lti = data['loan_amnt'] / data['person_income']
    ahf = data['person_age'] * data['cb_person_cred_hist_length']
    
    # Prepare data for AI
    input_df = pd.DataFrame([[
        data['person_income'], data['person_age'], data['loan_amnt'], lti, ahf
    ]], columns=features)
    
    prediction = model.predict(input_df)[0]
    result = "Risky" if prediction == 1 else "Safe"
    
    return jsonify({"status": result, "message": f"AI says this person is {result}, vro!"})

if __name__ == '__main__':
    app.run(port=5000)