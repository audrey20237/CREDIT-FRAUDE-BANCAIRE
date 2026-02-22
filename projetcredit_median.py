import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("creditcard.csv", sep=",",engine="python")

# Affiche les 5 premières lignes du dataset pour un aperçu rapide
print(df.head())

# Affiche le nombre de lignes et de colonnes du dataset
print(df.shape)

# Affiche un résumé des colonnes, types et valeurs manquante
print(df.info())

# Affiche les statistiques descriptives pour les colonnes numériques
print(df.describe())

# Compte le nombre d’occurrences de chaque classe 
print(df['Class'].value_counts())  

#Verifier le desequilibre des classes
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))

# Affichage de la distribution des classes
sns.countplot(x='Class', data=df)
plt.title("Répartition des classes (0=normal, 1=frauduleux)")
plt.show()

# Proportion des classes
print(df['Class'].value_counts(normalize=True))

# Supprimer les lignes avec valeurs manquantes
df = df.dropna()

# Encodage des colonnes catégorielles si présentes
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#Separer x(variable explicative) et y(cible ou variable à predire) 
X = df.drop("Class", axis=1)
y = df["Class"]

# Séparation train / test avant scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling uniquement sur le train 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

#  KNN sur un échantillon
# Pour éviter le blocage, on prend un échantillon de 5000 lignes
X_sample, _, y_sample, _ = train_test_split(
    X_train, y_train, train_size=5000, random_state=42, stratify=y_train
)

# Séparation train/test sur l'échantillon
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

# Tester KNN pour k = 1 à 10
knn_scores = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train_s)
    knn_scores.append(knn.score(X_test_s, y_test_s))

# Affichage graphique
plt.plot(range(1, 11), knn_scores, marker='o')
plt.title("Précision KNN sur échantillon selon k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

# Modele Random Forest 
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    class_weight='balanced',   
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Prédictions
y_pred_rf = rf.predict(X_test)

# Évaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest F1-score:", f1_score(y_test, y_pred_rf))
print("Matrice de confusion RF:\n", confusion_matrix(y_test, y_pred_rf))

# Sauvegarde
joblib.dump(rf, "credit_rf.pkl")
joblib.dump(scaler, "scaler.pkl")


# Modèle Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Entraîner le modèle
gb.fit(X_train, y_train)

# Prédictions sur le test
y_pred_gb = gb.predict(X_test)

# Évaluation
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting F1-score:", f1_score(y_test, y_pred_gb))
print("Matrice de confusion GB:\n", confusion_matrix(y_test, y_pred_gb))

# Sauvegarde
joblib.dump(gb, "credit_gb.pkl")

# Courbe ROC pour montrer la qualite globale de chaque modele à distinguer les classes
# Calculer les scores/probas pour chaque modèle
y_score_rf = rf.predict_proba(X_test)[:,1]  # probabilité pour la classe 1
y_score_gb = gb.predict_proba(X_test)[:,1]  # idem pour Gradient Boosting

# Courbes ROC pour chaque modèle
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_score_gb)

# Calcul AUC
auc_rf = roc_auc_score(y_test, y_score_rf)
auc_gb = roc_auc_score(y_test, y_score_gb)

# Tracer les courbes ROC
plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC={auc_gb:.3f})")
plt.plot([0,1],[0,1],'k--', label="Hasard")  # diagonale pour le hasard
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC - Comparaison des modèles")
plt.legend()
plt.grid()
plt.show()

# Graphique de l'importance des variables expliquatives sur le risaue lié au credit
# Extraire l'importance des features pour Gradient Boosting
importances = gb.feature_importances_
features = X.columns

# Trier par importance décroissante
indices = np.argsort(importances)[::-1]

# Tracer le graphique
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Importance des variables - Gradient Boosting")
plt.xlabel("Score d'importance")
plt.ylabel("Feature")
plt.show()