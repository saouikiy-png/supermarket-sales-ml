import pandas as pd
# 1 Lire le dataset
df = pd.read_csv("data/raw/sales.csv")
# 2 Afficher les premières lignes du dataset et des informations de base
print(df.head())  # pour vérifier que ça marche
print(df.info())         # types et colonnes
print(df.describe())     # stats numériques
print(df.isnull().sum())  # vérifier les valeurs manquantes

# 3 Analyser la variable cible
print(df["customer_type"].unique())
print(df["customer_type"].value_counts())
