import streamlit as st
import pandas as pd
import joblib
import numpy as np

# app.py
import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="💳 Détection de Fraude Carte Bancaire",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Titre et description
st.title("💳 Détection de Fraude aux Cartes Bancaires")
st.markdown("""
Ce modèle permet de prédire si une transaction est **normale** ou **frauduleuse**.
Entrez les caractéristiques de la transaction dans la sidebar et cliquez sur **Prédire**.
""")

# Charger les modèles et le scaler
rf_model = joblib.load("credit_rf.pkl")  # Random Forest
gb_model = joblib.load("credit_gb.pkl")  # Gradient Boosting si besoin
scaler = joblib.load("scaler.pkl")

# Sidebar pour entrer les valeurs des features
st.sidebar.header("💻 Entrer les caractéristiques de la transaction")

features = []
for i in range(1, 29):
    val = st.sidebar.number_input(f"V{i}", value=0.0, step=0.01, format="%.5f")
    features.append(val)

amount = st.sidebar.number_input("Montant de la transaction", value=0.0, step=0.01)
features.append(amount)

# Noms exacts des colonnes pour le scaler et les modèles
features_names = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# Vérification
if len(features) != len(features_names):
    st.error("Le nombre de features ne correspond pas au modèle.")
else:
    if st.sidebar.button("Prédire"):
        # Créer DataFrame avec colonnes correctes
        input_df = pd.DataFrame([features], columns=features_names)
        
        # Normalisation
        input_scaled = scaler.transform(input_df)
        
        # Prédiction avec Random Forest
        prediction = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0][1]
        
        # Affichage du résultat avec couleur
        if prediction == 1:
            st.error(f"🚨 Transaction suspecte ! Probabilité de fraude : {proba:.2%}")
        else:
            st.success(f"✅ Transaction normale. Probabilité de fraude : {proba:.2%}")

# Footer ou infos supplémentaires
st.markdown("---")
st.markdown("**Auteur :** Ton Nom  \n**Projet :** Détection de Fraude aux Cartes Bancaires")