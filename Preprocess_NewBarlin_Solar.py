# preprocess_solar_data.py

import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- Starting Solar Data Preprocessing ---")

# The URL for your messy, raw solar data file
url_solar_data = 'https://github.com/MeMeKhaingPhd/Unit-commitment-pyomo/blob/main/solar-reduced-berlin.csv'

# Define the output filename for the clean data
output_filename = 'C:\\Users\\Me Me Khaing\\Downloads\\Aswin Sir\\Data -- Berlin -- Oil and Solar\\solar_data_cleaned.csv'

try:
    df = pd.read_csv('Berlin_solar_regression.csv')
    print(f"\nDataset 'Berlin_solar_regression.csv' loaded successfully.")
    print(f"Shape of the dataset: {df.shape}")
except FileNotFoundError:
    print("\nError: 'Berlin_solar_regression.csv' not found.")
    print("Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"\nAn error occurred while loading the CSV: {e}")
    exit()


print("\n--- Data Exploration ---")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())


print("\n--- Preprocessing ---")
target_column = 'X50Hertz..MW.'

if target_column not in df.columns:
    print(f"\nError: Target column '{target_column}' not found.")
    print(f"Available columns: {list(df.columns)}")
    exit()
else:
    print(f"Identified Target Variable: '{target_column}'")


datetime_cols = df.select_dtypes(include=['datetime', 'datetime64', 'object']).columns
potential_dt_col = None
for col in datetime_cols:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].iloc[0])
            potential_dt_col = col
            print(f"Potential datetime column found (type object): '{col}'")
            break
        except (ValueError, TypeError):
            continue
    elif 'datetime' in str(df[col].dtype):
        potential_dt_col = col
        print(f"Datetime column found: '{col}'")
        break

is_time_series = False
if potential_dt_col:
    print(f"Converting '{potential_dt_col}' to datetime objects...")
    try:
        df[potential_dt_col] = pd.to_datetime(df[potential_dt_col])
        print(f"Sorting data by '{potential_dt_col}' for time series analysis.")
        df = df.sort_values(by=potential_dt_col).reset_index(drop=True)
        print("Extracting time-based features (hour, dayofyear, month, year, dayofweek)...")
        df['hour'] = df[potential_dt_col].dt.hour
        df['dayofyear'] = df[potential_dt_col].dt.dayofyear
        df['month'] = df[potential_dt_col].dt.month
        df['year'] = df[potential_dt_col].dt.year
        df['dayofweek'] = df[potential_dt_col].dt.dayofweek
        print(f"Time features extracted. Original datetime column '{potential_dt_col}' kept for now.")
        is_time_series = True
    except Exception as e:
        print(f"Could not convert '{potential_dt_col}' or extract features: {e}")


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if target_column in numeric_cols:
    numeric_cols.remove(target_column)

if df[numeric_cols].isnull().sum().sum() > 0:
    print("Handling missing values using median imputation for numeric features...")
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f" - Imputed NaNs in '{col}' with median value: {median_val:.4f}")
else:
    print("No missing values found in numeric feature columns.")


features = [col for col in df.select_dtypes(include=np.number).columns if col != target_column]
non_numeric_features = df.select_dtypes(exclude=np.number).columns
if potential_dt_col and potential_dt_col in non_numeric_features:
     non_numeric_features = non_numeric_features.drop(potential_dt_col)

if len(non_numeric_features) > 0:
    print(f"\nWarning: Non-numeric columns found: {list(non_numeric_features)}")
    print("Excluding non-numeric columns from features. Consider encoding them.")
else:
     print("\nAll feature columns used are numeric.")
# 6. Save the cleaned DataFrame to a new CSV file
try:
    print(f"\nSaving cleaned data to '{output_filename}'...")
    # We save the index because it contains our important Timestamp
    df.to_csv(output_filename, index=True)
    print(f"Successfully created '{output_filename}'.")
    print(f"Final shape of cleaned data: {df.shape}")
except Exception as e:
    print(f"\nFATAL ERROR: Could not save the cleaned data file. Error: {e}")
    exit()

print("\n--- Preprocessing Complete! ---")