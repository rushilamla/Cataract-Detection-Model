import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset (Update with actual path)
data_dir = r"C:\Users\rushi\Downloads\Dataset"  # Change this to your dataset folder
categories = ["Healthy", "Cataract"]
X, y = [], []

# Image Preprocessing
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        image = cv2.resize(image, (128, 128))  # Resize
        image = image.flatten()  # Flatten image
        X.append(image)
        y.append(label)

X = np.array(X) / 255.0  # Normalize pixel values
y = np.array(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "cataract_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training complete. Logistic Regression model saved as 'cataract_model.pkl'")
