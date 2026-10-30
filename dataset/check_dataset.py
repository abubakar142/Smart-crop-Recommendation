import pandas as pd

# Load the crop dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display dataset size
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumn Names:")
print(data.columns)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Display available crops
print("\nAvailable Crops:")
print(data["label"].unique())

print("\nDataset loaded successfully!")