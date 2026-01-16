import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to processed dataset
DATA_PATH = "data/processed/data_prepared.csv"

# Check if file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check if dataset is empty
if df.empty:
    raise ValueError(
        
    )

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
