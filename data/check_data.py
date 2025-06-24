import pandas as pd

# Load the dataset
print("Loading creditcard.csv...")
df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Normal transactions: {sum(df['Class'] == 0)}")
print(f"Fraudulent transactions: {sum(df['Class'] == 1)}")
print(f"Fraud percentage: {(sum(df['Class'] == 1) / len(df)) * 100:.3f}%")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset loaded successfully! ✅")

input("Press Enter to continue...")  # Add this line