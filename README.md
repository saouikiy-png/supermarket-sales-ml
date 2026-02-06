# supermarket-sales-ml


### **Titre du projet**
Analyse et prédiction du type de client d’un supermarché à partir des données de ventes

### **Objectif**
Analyser le comportement des clients d’un supermarché et prédire leur type (Member / Normal) afin d’aider à la prise de décision marketing.

### **Contexte / problème**
Le supermarché dispose de données transactionnelles mais ne sait pas comment les exploiter pour comprendre ses clients ni anticiper leur comportement.
Le problème est d’extraire des insights utiles et de construire un modèle capable de classifier les clients selon leur profil.

### **Stack / outils**
* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Jupyter Notebook

### **Ce que j’ai fait **
* Nettoyage et préparation des données (valeurs manquantes, encodage des variables catégorielles)
* Analyse exploratoire des données (EDA) pour identifier les tendances et corrélations
* Construction de modèles de classification (Régression Logistique, Naive Bayes)
* Évaluation des modèles avec des métriques (accuracy, confusion matrix)
* Comparaison des performances pour choisir le modèle le plus pertinent

### **Difficulté rencontrée + solution**
* **Difficulté** : Les variables catégorielles ne pouvaient pas être utilisées directement par les modèles
* **Solution** : Application du One-Hot Encoding pour transformer les données en format exploitable

### **Résultat )**
* Modèle de classification atteignant une accuracy d’environ **51 %**
* Identification de variables clés influençant le type de client
* Modèle final capable de prédire le type de client à partir des données de vente

