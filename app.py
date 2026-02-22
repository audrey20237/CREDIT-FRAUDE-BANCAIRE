import streamlit as st
import pandas as pd
import joblib
import numpy as np

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(
    page_title="💳 Détection Fraude Carte Bancaire",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar menu
# -----------------------------
st.sidebar.title("Menu")
section = st.sidebar.radio("Aller à :", ["Accueil", "Analyse de données", "Prédiction"])

# -----------------------------
# Section Accueil
# -----------------------------
if section == "Accueil":
    st.title("💳 Projet Détection de Fraude aux Credits Bancaires")
    st.markdown("""
    **Auteur : LUCRECE ATANGANA  

    Bienvenue sur cette application interactive de détection de fraude aux credits bancaires.  
    Vous pouvez explorer les données, visualiser des graphiques et prédire la probabilité qu'une transaction soit frauduleuse.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    st.markdown("---")

# -----------------------------
# Section Analyse de données
# -----------------------------
elif section == "Analyse de données":
    st.title("📊 Analyse des données")
    
    # Charger dataset
    df = pd.read_csv("creditcard.csv")
    
    # Affichage aperçu
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())
    
    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    
    # Distribution des classes
    st.subheader("Répartition des classes")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    ax.set_title("Transactions normales vs frauduleuses")
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Carte de corrélation")
    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# Section Prédiction
# -----------------------------
elif section == "Prédiction":
    st.title("🤖 Prédiction de fraude")
    st.markdown("Entrez les caractéristiques de la transaction dans la sidebar et cliquez sur **Prédire**.")

    # Charger modèles et scaler
    rf_model = joblib.load("credit_rf.pkl")
    gb_model = joblib.load("credit_gb.pkl")  # optionnel
    scaler = joblib.load("scaler.pkl")

    st.sidebar.header("Entrer les caractéristiques")

    # -----------------------------
    # Sidebar avec colonnes pour les 29 inputs
    # -----------------------------
    # Créer 3 colonnes pour les 28 V + 1 Amount
    num_features = 28
    input_values = []
    cols = st.sidebar.columns(3)
    for i in range(1, num_features+1):
        col = cols[i % 3]
        val = col.number_input(f"V{i}", value=0.0, step=0.01, format="%.5f")
        input_values.append(val)
    
    amount = st.sidebar.number_input("Montant", value=0.0, step=0.01)
    input_values.append(amount)

    # -----------------------------
    # Prédiction avec le bouton
    # -----------------------------
    if st.sidebar.button("Prédire"):
        # Utiliser les noms exacts du scaler pour éviter l'erreur
        features_names = scaler.feature_names_in_
        input_df = pd.DataFrame([input_values], columns=features_names)
        input_scaled = scaler.transform(input_df)

        # Prédiction avec Random Forest
        prediction = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0][1]

        # Affichage du résultat
        if prediction == 1:
            st.error(f"🚨 Transaction suspecte ! Probabilité : {proba:.2%}")
        else:
            st.success(f"✅ Transaction normale. Probabilité : {proba:.2%}")

        # Graphique jauge circulaire pour probabilité
        st.subheader("Probabilité de fraude")
        st.write("La jauge indique la probabilité que la transaction soit frauduleuse :")
        fig3, ax3 = plt.subplots(figsize=(4,4))
        ax3.pie([proba, 1-proba], labels=["Fraude", "Normale"], colors=["red","green"], autopct="%1.1f%%", startangle=90)
        ax3.axis("equal")
        st.pyplot(fig3)