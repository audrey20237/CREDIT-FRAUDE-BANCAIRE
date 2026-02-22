
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score

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
    st.markdown("Entrez les caractéristiques de la transaction dans la sidebar et cliquez sur **Prédire**.")

    # Charger modèles et scaler
    try:
        rf_model = joblib.load("credit_rf.pkl")
    except FileNotFoundError:
        st.error("Le fichier credit_rf.pkl est introuvable.")
        st.stop()

    try:
        gb_model = joblib.load("credit_gb.pkl")
    except FileNotFoundError:
        gb_model = None  # optionnel

    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("Le fichier scaler.pkl est introuvable.")
        st.stop()

    # Sidebar pour 28 features + Montant
    st.sidebar.header("Entrer les caractéristiques")
    num_features = 28
    input_values = []
    cols = st.sidebar.columns(3)
    for i in range(1, num_features + 1):
        col = cols[(i - 1) % 3]
        val = col.number_input(f"V{i}", value=0.0, step=0.01, format="%.5f")
        input_values.append(val)
    amount = st.sidebar.number_input("Montant", value=0.0, step=0.01)
    input_values.append(amount)

    # Bouton Prédire
    if st.sidebar.button("Prédire"):
        # Déterminer les colonnes attendues par le scaler
        if hasattr(scaler, "feature_names_in_"):
            expected_columns = list(scaler.feature_names_in_)
        else:
            expected_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]

        if len(expected_columns) != len(input_values):
            st.error("Le nombre de colonnes du scaler ne correspond pas au nombre d'inputs. Vérifie l'ordre des features.")
            st.stop()

        # Créer DataFrame et scaler
        input_df = pd.DataFrame([input_values], columns=expected_columns)
        input_scaled = scaler.transform(input_df)

        # Prédiction Random Forest
        prediction = rf_model.predict(input_scaled)[0]
        try:
            proba = rf_model.predict_proba(input_scaled)[0][1]
        except AttributeError:
            proba = None

        # Affichage du résultat
        if prediction == 1:
            st.markdown(
                f"<h3 style='color:red;'>🚨 Transaction suspecte ! Probabilité : "
                f"{proba:.2%}</h3>" if proba is not None else "<h3 style='color:red;'>🚨 Transaction suspecte !</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h3 style='color:green;'>✅ Transaction normale. Probabilité : "
                f"{proba:.2%}</h3>" if proba is not None else "<h3 style='color:green;'>✅ Transaction normale.</h3>",
                unsafe_allow_html=True
            )

        # Jauge interactive
        if proba is not None:
            fig_gauge = px.pie(
                names=["Fraude", "Normale"],
                values=[proba, 1 - proba],
                color_discrete_sequence=["red", "green"],
                hole=0.6
            )
            fig_gauge.update_layout(
                showlegend=True,
                title_text="Probabilité de fraude",
                annotations=[dict(text=f"{proba:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)]
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