
# Create a "Debt-to-Income" ratio (This is what real bank AIs look at!)
df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

# Create a "Risk Score" based on age and history
df['age_history_factor'] = df['person_age'] * df['cb_person_cred_hist_length']

# Update your Features to include these complex math columns
X = df[['person_income', 'person_age', 'loan_amnt', 'loan_to_income_ratio', 'age_history_factor']]
