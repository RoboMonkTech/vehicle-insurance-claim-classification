import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

DATA_PATH = "dataset/vehicle_insurance.csv"

try:
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print(" VEHICLE INSURANCE CLAIM DATASET ANALYSIS ")
    print("=" * 70)

    print("\nDataset Loaded Successfully!")

    print("\nShape of Dataset")
    print(df.shape)

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nFirst Five Rows")
    print(df.head())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nTarget Distribution")
    print(df["Response"].value_counts())

    print("\nTarget Distribution (%)")
    print(round(df["Response"].value_counts(normalize=True) * 100, 2))

    print("\nSummary Statistics")
    print(df.describe())

except FileNotFoundError:
    print("Dataset not found!")

except Exception as e:
    print("Error:", e)