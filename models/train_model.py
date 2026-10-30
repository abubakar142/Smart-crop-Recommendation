import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# 1. Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# 2. Separate input and output
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINING COMPLETED")
print("Accuracy:", round(accuracy * 100, 2), "%")

# 8. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Save trained model
joblib.dump(model, "models/crop_model.pkl")
print("\nModel saved successfully!")