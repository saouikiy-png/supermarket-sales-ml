import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to processed dataset
DATA_PATH = "data/processed/data_prepared.csv"

# Check if file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Le fichier data_prepared.csv est introuvable. Vérifie le preprocessing.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check if dataset is empty
if df.empty:
    raise ValueError("Le fichier data_prepared.csv est vide. Attends que Personne A le remplisse.")

# Separate features and target
TARGET_COLUMN = "target"  
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Train / Test split (80/20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Models setup terminé avec succès.")
print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logistic Regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)  
y_pred_nb = nb.predict(X_test)

import csv

results_file = "results/metrics.csv"


if not os.path.exists("results"):
    os.makedirs("results")

# Metrics dictionary
results = [
    ["Model", "Accuracy", "Precision", "Recall", "F1-score"],
    [
        "Logistic Regression",
        accuracy_score(y_test, y_pred_logreg),
        precision_score(y_test, y_pred_logreg, zero_division=0),
        recall_score(y_test, y_pred_logreg, zero_division=0),
        f1_score(y_test, y_pred_logreg, zero_division=0),
    ],
    [
        "Naive Bayes",
        accuracy_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_nb, zero_division=0),
        recall_score(y_test, y_pred_nb, zero_division=0),
        f1_score(y_test, y_pred_nb, zero_division=0),
    ],
]

with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(f"Models trained and results saved to {results_file}")
