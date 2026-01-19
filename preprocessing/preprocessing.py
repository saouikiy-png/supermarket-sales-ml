import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_csv("data/raw/sales.csv")

# Séparer la variable cible
y = df["customer_type"]
X = df.drop(columns=["customer_type", "sale_id", "product_name"])

# Colonnes numériques
num_cols = [
    "unit_price",
    "quantity",
    "tax",
    "total_price",
    "reward_points"
]

# Normalisation
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Encodage One-Hot des variables catégorielles
X = pd.get_dummies(
    X,
    columns=["branch", "city", "gender", "product_category"],
    drop_first=True
)

# Recombiner X et y
df_final = pd.concat([X, y], axis=1)
print(df_final.shape)
print(df_final.head())
# Sauvegarder le dataset final
import os

os.makedirs("data/processed", exist_ok=True)
df_final.to_csv("data/processed/data_prepared.csv", index=False)

print("Dataset final prêt pour le ML")


