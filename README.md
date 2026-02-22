 💳 Détection de Fraude aux Cartes Bancaires

 📌 Présentation du projet
Ce projet vise à détecter les transactions frauduleuses sur les cartes bancaires en utilisant le Machine Learning.

Le dataset utilisé contient des caractéristiques anonymisées des transactions (V1–V28) ainsi que le montant de la transaction.

 ⚠️ Challenge
Le dataset est fortement déséquilibré :
- 99,8 % transactions normales
- 0,2 % transactions frauduleuses

Ainsi, la précision seule (accuracy) n’est pas un indicateur fiable.

🤖 Modèles utilisés
- Random Forest
- Gradient Boosting

 📊 Métriques d’évaluation
- F1-score
- ROC-AUC
- Matrice de confusion

 🚀 Déploiement
Le modèle est déployé via Streamlit.

 🌐 Démo en ligne
Vous pouvez tester l’application ici : [Application Streamlit](https://share.streamlit.io/TON_UTILISATEUR/credit-fraud-detection/app.py)

 ⚡ Installation
1. Cloner le dépôt :
```bash
git clone https://github.com/TON_UTILISATEUR/credit-fraud-detection.git