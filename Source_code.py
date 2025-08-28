# Wine Quality Prediction Project A

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv("WineQT.csv")

# Step 3: Drop 'Id' column (not useful)
df = df.drop(columns=["Id"])

# Step 4: Basic Info
print("Shape of dataset:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Step 5: Distribution of Wine Quality
plt.figure(figsize=(6,4))
sns.countplot(x="quality", hue="quality", data=df, palette="viridis", legend=False)
plt.title("Wine Quality Distribution")
plt.show()

# Step 6: Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Feature vs Quality Boxplots
features = ["alcohol", "sulphates", "citric acid", "volatile acidity"]
plt.figure(figsize=(12,8))
for i, feature in enumerate(features, 1):
    plt.subplot(2,2,i)
    sns.boxplot(x="quality", y=feature, hue="quality", data=df, palette="Set2", legend=False)
    plt.title(f"{feature} vs Quality")
plt.tight_layout()
plt.show()

# ------------------------
# 🔹 Step 8: Data Preprocessing
# ------------------------
X = df.drop("quality", axis=1)  # features
y = df["quality"]               # target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------
# 🔹 Step 9: Train Models
# ------------------------

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# SGD Classifier
sgd = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd.fit(X_train_scaled, y_train)
sgd_pred = sgd.predict(X_test_scaled)

# Support Vector Classifier
svc = SVC(random_state=42)
svc.fit(X_train_scaled, y_train)
svc_pred = svc.predict(X_test_scaled)

# ------------------------
# 🔹 Step 10: Evaluation
# ------------------------

def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate all models
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("SGD Classifier", y_test, sgd_pred)
evaluate_model("Support Vector Classifier", y_test, svc_pred)

# ------------------------
# 🔹 Step 11: Feature Importance (Random Forest)
# ------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(
    x=importances[indices], 
    y=X.columns[indices], 
    hue=X.columns[indices], 
    palette="viridis", 
    legend=False, dodge=False
)
plt.title("Feature Importance (Random Forest)")
plt.show()
