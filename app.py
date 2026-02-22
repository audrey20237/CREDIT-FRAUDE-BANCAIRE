
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
    rf_model = joblib.load("credit_rf.pkl")
    gb_model = joblib.load("credit_gb.pkl")  # optionnel
    scaler = joblib.load("scaler.pkl")

    st.sidebar.header("Entrer les caractéristiques")

    # Sidebar pour 28 features + Montant
    num_features = 28
    input_values = []
    cols = st.sidebar.columns(3)
    for i in range(1, num_features+1):
        col = cols[i % 3]
        val = col.number_input(f"V{i}", value=0.0, step=0.01, format="%.5f")
        input_values.append(val)
    amount = st.sidebar.number_input("Montant", value=0.0, step=0.01)
    input_values.append(amount)

    # Bouton Prédire
    if st.sidebar.button("Prédire"):
        # Créer DataFrame pour scaler
        input_df = pd.DataFrame([input_values], columns=scaler.feature_names_in_)
        input_scaled = scaler.transform(input_df)
        
        # Prédiction Random Forest
        prediction = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0][1]
        
        # Affichage du résultat
        if prediction == 1:
            st.markdown(f"<h3 style='color:red;'>🚨 Transaction suspecte ! Probabilité : {proba:.2%}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:green;'>✅ Transaction normale. Probabilité : {proba:.2%}</h3>", unsafe_allow_html=True)
        
        # Jauge circulaire interactive
        fig_gauge = px.pie(
            names=["Fraude", "Normale"],
            values=[proba, 1-proba],
            color_discrete_sequence=["red","green"],
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

    # Charger dataset complet ou échantillon
    df_eval = pd.read_csv("creditcard_sample.csv")
    X_eval = df_eval.drop("Class", axis=1)
    y_eval = df_eval["Class"]
    
    # Normalisation
    X_scaled = scaler.transform(X_eval)
    
    # Prédiction modèles
    y_pred_rf = rf_model.predict(X_scaled)
    y_pred_gb = gb_model.predict(X_scaled)

    # Metrics Random Forest
    st.subheader("Random Forest")
    st.text(f"Accuracy : {accuracy_score(y_eval, y_pred_rf):.3f}")
    st.text(f"F1-score : {f1_score(y_eval, y_pred_rf):.3f}")
    st.text(f"ROC-AUC : {roc_auc_score(y_eval, rf_model.predict_proba(X_scaled)[:,1]):.3f}")
    st.text("Classification Report :\n" + classification_report(y_eval, y_pred_rf))
    
    # Metrics Gradient Boosting
    st.subheader("Gradient Boosting")
    st.text(f"Accuracy : {accuracy_score(y_eval, y_pred_gb):.3f}")
    st.text(f"F1-score : {f1_score(y_eval, y_pred_gb):.3f}")
    st.text(f"ROC-AUC : {roc_auc_score(y_eval, gb_model.predict_proba(X_scaled)[:,1]):.3f}")
    st.text("Classification Report :\n" + classification_report(y_eval, y_pred_gb))

