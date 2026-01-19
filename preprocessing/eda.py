import pandas as pd
import matplotlib.pyplot as plt
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
# Graphique 1 — Répartition des types de clients
counts = df["customer_type"].value_counts()
plt.figure()
counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Répartition des types de clients")
plt.ylabel("")
plt.show()
# Graphique 2 — Parts de marché par catégorie de produit
category_sales = df.groupby("product_category")["total_price"].sum()
plt.figure()
category_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("Parts de marché par catégorie de produit")
plt.ylabel("")
plt.show()
#Graphique 3 — Distribution des variables numériques
plt.figure()
df["unit_price"].hist()
plt.title("Distribution du prix unitaire")
plt.xlabel("Prix unitaire")
plt.ylabel("Fréquence")
plt.show()

plt.figure()
df["total_price"].hist()
plt.title("Distribution du montant total")
plt.xlabel("Montant total")
plt.ylabel("Fréquence")
plt.show()

