import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# 1. Display dataset information
print("DATASET INFORMATION")
print(data.info())

# 2. Display first five rows
print("\nFIRST FIVE ROWS")
print(data.head())

# 3. Check dataset size
print("\nDATASET SHAPE")
print(data.shape)

# 4. Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# 5. Statistical analysis
print("\nSTATISTICAL SUMMARY")
print(data.describe())

# 6. Count each crop
print("\nCROP DISTRIBUTION")
print(data["label"].value_counts())

# 7. Crop distribution graph
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x="label")
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.tight_layout()
plt.savefig("reports/crop_distribution.png")
plt.show()

# 8. Soil nutrient analysis
plt.figure(figsize=(10, 5))
data[["N", "P", "K"]].hist(figsize=(10, 6))
plt.suptitle("Soil Nutrient Distribution")
plt.tight_layout()
plt.savefig("reports/soil_nutrients.png")
plt.show()

# 9. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    data.drop(columns=["label"]).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("reports/correlation.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")