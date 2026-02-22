
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve

# -----------------------------
# Configuration page
# -----------------------------
st.set_page_config(
    page_title="💳 Détection Fraude Carte Bancaire",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar menu
# -----------------------------
st.sidebar.title("Menu")
section = st.sidebar.radio("Navigation :", ["Accueil", "Analyse de données", "Prédiction", "Évaluation"])

# -----------------------------
# Section Accueil
# -----------------------------
if section == "Accueil":
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>💳 Projet Détection de Fraude Carte Bancaire</h1>", unsafe_allow_html=True)
    st.markdown("**Auteur : LUCRECE ATANGANA**")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    
    st.markdown("### Avant-propos")
    st.markdown("""
    Les fraudes aux cartes bancaires représentent un problème majeur pour les institutions financières.
    Ce projet vise à détecter automatiquement les transactions suspectes grâce à des modèles de Machine Learning.
    L’application permet à la fois d’explorer les données et de prédire le risque de fraude pour une transaction donnée.
    """)

    st.markdown("### Méthodes utilisées")
    st.markdown("""
    - **Random Forest** : un ensemble d'arbres de décision pour améliorer la robustesse et la précision.  
    - **Gradient Boosting** : méthode d’ensemblage qui corrige les erreurs des arbres précédents.  
    """)

# -----------------------------
# Section Analyse de données
# -----------------------------
elif section == "Analyse de données":
    st.markdown("<h2 style='color:#FF8C00;'>📊 Analyse des données</h2>", unsafe_allow_html=True)
    
    # Charger dataset échantillon local
    df = pd.read_csv("creditcard_sample.csv")
    
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())
    
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    
    st.subheader("Répartition des classes")
    fig = px.histogram(df, x="Class", color="Class", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Carte de corrélation")
    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# Section Prédiction
# -----------------------------
elif section == "Prédiction":
    st.markdown("<h2 style='color:#32CD32;'>🤖 Prédiction de fraude</h2>", unsafe_allow_html=True)
    st.markdown("Entrez les caractéristiques de la transaction dans la sidebar puis cliquez sur **Prédire**.")

    # -------------------------
    # Chargement modèles / scaler
    # -------------------------
    try:
        rf_model = joblib.load("credit_rf.pkl")
    except FileNotFoundError:
        st.error("Le fichier credit_rf.pkl est introuvable.")
        st.stop()

    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("Le fichier scaler.pkl est introuvable.")
        st.stop()

    # -------------------------
    # Déterminer automatiquement les colonnes attendues
    # -------------------------
    if hasattr(scaler, "feature_names_in_"):
        expected_columns = list(scaler.feature_names_in_)
    else:
        # fallback sécurisé si le scaler a été entraîné sur numpy array
        expected_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    st.sidebar.header("Entrer les caractéristiques")

    # -------------------------
    # Création dynamique des inputs selon les colonnes attendues
    # -------------------------
    input_dict = {}

    cols = st.sidebar.columns(3)

    for i, col_name in enumerate(expected_columns):
        col = cols[i % 3]

        # Valeur par défaut intelligente
        default_value = 0.0

        input_value = col.number_input(
            label=col_name,
            value=default_value,
            step=0.01,
            format="%.5f"
        )

        input_dict[col_name] = input_value

    # -------------------------
    # Section prédiction
    # -------------------------
    if st.sidebar.button("Prédire"):

        # Création DataFrame aligné automatiquement
        input_df = pd.DataFrame([input_dict])

        # Sécurisation : réordonner EXACTEMENT comme le scaler
        input_df = input_df[expected_columns]

        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.error("Erreur lors du scaling. Vérifie que les features correspondent au modèle.")
            st.stop()

        # -------------------------
        # Prédiction
        # -------------------------
        prediction = rf_model.predict(input_scaled)[0]

        try:
            proba = rf_model.predict_proba(input_scaled)[0][1]
        except:
            proba = None

        # -------------------------
        # Affichage résultat
        # -------------------------
        if prediction == 1:
            if proba is not None:
                st.markdown(
                    f"<h3 style='color:red;'>🚨 Transaction suspecte ! Probabilité : {proba:.2%}</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h3 style='color:red;'>🚨 Transaction suspecte !</h3>",
                    unsafe_allow_html=True
                )
        else:
            if proba is not None:
                st.markdown(
                    f"<h3 style='color:green;'>✅ Transaction normale. Probabilité : {proba:.2%}</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h3 style='color:green;'>✅ Transaction normale.</h3>",
                    unsafe_allow_html=True
                )

        # -------------------------
        # Visualisation probabilité
        # -------------------------
        if proba is not None:
            fig_gauge = px.pie(
                names=["Fraude", "Normale"],
                values=[proba, 1 - proba],
                hole=0.6,
                color_discrete_sequence=["red", "green"]
            )

            fig_gauge.update_layout(
                title="Probabilité de fraude",
                annotations=[
                    dict(text=f"{proba:.1%}", x=0.5, y=0.5, font_size=22, showarrow=False)
                ]
            )

            st.plotly_chart(fig_gauge, use_container_width=True)


# -----------------------------
# Section Évaluation des modèles
# -----------------------------
elif section == "Évaluation":
    st.markdown("<h2 style='color:#800080;'>📈 Évaluation des modèles</h2>", unsafe_allow_html=True)

    # Charger modèles et scaler
    try:
        rf_model = joblib.load("credit_rf.pkl")
    except FileNotFoundError:
        st.error("Le fichier credit_rf.pkl est introuvable.")
        st.stop()

    try:
        gb_model = joblib.load("credit_gb.pkl")
    except FileNotFoundError:
        gb_model = None

    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("Le fichier scaler.pkl est introuvable.")
        st.stop()

    # Charger dataset
    df_eval = pd.read_csv("creditcard_sample.csv")
    X_eval = df_eval.drop("Class", axis=1)
    y_eval = df_eval["Class"]

    # Vérifier alignement colonnes scaler
    if hasattr(scaler, "feature_names_in_"):
        expected_columns = list(scaler.feature_names_in_)
        if not all(col in X_eval.columns for col in expected_columns):
            st.error("Les colonnes du dataset ne correspondent pas à celles du scaler.")
            st.stop()
        X_eval = X_eval[expected_columns]
    else:
        # supposons que X_eval contient les colonnes correctes
        pass

    # Normalisation
    X_scaled = scaler.transform(X_eval)

    # Prédictions
    y_pred_rf = rf_model.predict(X_scaled)
    if gb_model is not None:
        y_pred_gb = gb_model.predict(X_scaled)

    # Metrics Random Forest
    st.subheader("Random Forest")
    st.text(f"Accuracy : {accuracy_score(y_eval, y_pred_rf):.3f}")
    st.text(f"F1-score : {f1_score(y_eval, y_pred_rf):.3f}")
    st.text(f"ROC-AUC : {roc_auc_score(y_eval, rf_model.predict_proba(X_scaled)[:,1]):.3f}")
    st.text("Classification Report :\n" + classification_report(y_eval, y_pred_rf))

    # Metrics Gradient Boosting
    if gb_model is not None:
        st.subheader("Gradient Boosting")
        st.text(f"Accuracy : {accuracy_score(y_eval, y_pred_gb):.3f}")
        st.text(f"F1-score : {f1_score(y_eval, y_pred_gb):.3f}")
        try:
            roc_auc_gb = roc_auc_score(y_eval, gb_model.predict_proba(X_scaled)[:,1])
        except AttributeError:
            roc_auc_gb = None
        st.text(f"ROC-AUC : {roc_auc_gb:.3f}" if roc_auc_gb is not None else "ROC-AUC : N/A")
        st.text("Classification Report :\n" + classification_report(y_eval, y_pred_gb))

st.subheader("Courbe ROC")

# Probabilités
y_score_rf = rf_model.predict_proba(X_scaled)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_eval, y_score_rf)
auc_rf = roc_auc_score(y_eval, y_score_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")

# Gradient Boosting si disponible
if gb_model is not None:
    try:
        y_score_gb = gb_model.predict_proba(X_scaled)[:, 1]
        fpr_gb, tpr_gb, _ = roc_curve(y_eval, y_score_gb)
        auc_gb = roc_auc_score(y_eval, y_score_gb)
        plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC={auc_gb:.3f})")
    except:
        pass

plt.plot([0,1],[0,1],'k--', label="Hasard")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC - Comparaison des modèles")
plt.legend()
plt.grid()

st.pyplot(plt)

st.subheader("Importance des variables (Gradient Boosting)")

if gb_model is not None:
    try:
        importances = gb_model.feature_importances_
        
        # Récupérer noms des features correctement
        if hasattr(scaler, "feature_names_in_"):
            features = scaler.feature_names_in_
        else:
            features = X_eval.columns

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10,6))
        sns.barplot(
            x=importances[indices],
            y=np.array(features)[indices]
        )
        plt.title("Importance des variables - Gradient Boosting")
        plt.xlabel("Score d'importance")
        plt.ylabel("Feature")

        st.pyplot(plt)
    except:
        st.warning("Impossible d'afficher l'importance des variables.")