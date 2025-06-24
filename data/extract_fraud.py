import pandas as pd

# Load the dataset
df = pd.read_csv('data\creditcard.csv')

# Extract a single fraudulent transaction
fraud_transaction = df[df['Class'] == 1].iloc[0]  # Get the first fraud case

# Drop the 'Class' column for testing
test_transaction = fraud_transaction.drop('Class')

print(test_transaction)