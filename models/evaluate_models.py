from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Path to metrics file
METRICS_FILE = "results/metrics.csv"

# Lire les metrics
df = pd.read_csv(METRICS_FILE)

# Créer le dossier pour les figures si non existant
FIGURES_DIR = "results/figures"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

print("=== Résultats des modèles ===\n")
print(df.to_string(index=False))

# Identifier le meilleur modèle pour chaque métrique
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
best_models = {metric: df.loc[df[metric].idxmax()]['Model']
               for metric in metrics}

print("\n=== Meilleur modèle par métrique ===")
for metric, model in best_models.items():
    print(f"{metric}: {model}")

# Ajouter classement automatique pour chaque métrique
print("\n=== Classement des modèles par métrique ===")
for metric in metrics:
    ranked = df[['Model', metric]].sort_values(by=metric, ascending=False)
    print(f"\nTop {metric}:")
    print(ranked.to_string(index=False))

# Recommandation finale
print("\n=== Recommandation générale ===")
if len(set(best_models.values())) == 1:
    print(
        f"Le modèle recommandé est {list(best_models.values())[0]}, il est performant sur toutes les métriques.")
else:
    print("Il n'y a pas un modèle parfait pour toutes les métriques. Il faut choisir selon la priorité du projet.")

# Confusion Matrix - Logistic Regression
y_true = pd.read_csv("data/processed/data_prepared.csv")["customer_type"]

# Recharger les prédictions depuis les métriques
# (on relance les modèles pour la matrice)

df_full = pd.read_csv("data/processed/data_prepared.csv")

X = df_full.drop("customer_type", axis=1)
y = df_full["customer_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_logreg, labels=["Member", "Normal"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Member", "Normal"])
disp.plot(cmap="Blues")

plt.title("Matrice de confusion – Logistic Regression")
plt.savefig(os.path.join(
    FIGURES_DIR, "confusion_matrix_logistic_regression.png"))
plt.close()

print("Matrice de confusion sauvegardée pour Logistic Regression.")

# Création des figures avec mise en valeur du meilleur modèle et tri
for metric, default_color in zip(metrics, ["skyblue", "lightgreen", "salmon", "orchid"]):
    # Trier le DataFrame pour ce métrique (du meilleur au moins bon)
    df_sorted = df.sort_values(
        by=metric, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(6, 4))

    # Couleurs: couleur spéciale pour le meilleur modèle
    colors = [default_color if model != best_models[metric]
              else "gold" for model in df_sorted['Model']]

    plt.bar(df_sorted['Model'], df_sorted[metric], color=colors)
    plt.title(f"{metric} par modèle")
    plt.ylabel(metric)
    plt.ylim(0, 1)

    # Ajouter la valeur au-dessus de chaque barre
    for i, value in enumerate(df_sorted[metric]):
        plt.text(i, value + 0.02, f"{value:.2f}", ha='center')

    plt.savefig(os.path.join(FIGURES_DIR, f"{metric.lower()}.png"))
    plt.close()

print(f"\nLes figures ont été sauvegardées dans {FIGURES_DIR}")
