# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    accuracy_score,
    roc_curve
)

# -----------------------------
# Configuration page
# -----------------------------
st.set_page_config(
    page_title="💳 Détection Fraude Carte Bancaire",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Cache (IMPORTANT pour Streamlit Cloud)
# -----------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("credit_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        gb = joblib.load("credit_gb.pkl")
    except:
        gb = None
    return rf, gb, scaler


@st.cache_data
def load_data():
    return pd.read_csv("creditcard_sample.csv")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Menu")
section = st.sidebar.radio(
    "Navigation :",
    ["Accueil", "Analyse de données", "Prédiction", "Évaluation"]
)

# =============================
# ACCUEIL
# =============================
if section == "Accueil":

    st.title("💳 Détection de Fraude Carte Bancaire")
    st.markdown("**Auteur : LUCRECE ATANGANA**")

    st.markdown("""
    Cette application permet :
    - L'exploration des données
    - La prédiction de fraude
    - L'évaluation des modèles ML
    """)

# =============================
# ANALYSE
# =============================
elif section == "Analyse de données":

    st.header("📊 Analyse des données")

    df = load_data()

    st.subheader("Aperçu")
    st.dataframe(df.head())

    st.subheader("Statistiques")
    st.write(df.describe())

    st.subheader("Répartition des classes")
    fig = px.histogram(df, x="Class", color="Class")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de corrélation")
    fig2, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

# =============================
# PRÉDICTION
# =============================
elif section == "Prédiction":

    st.header("🤖 Prédiction de fraude")

    rf_model, _, scaler = load_models()

    # Colonnes attendues
    if hasattr(scaler, "feature_names_in_"):
        expected_columns = list(scaler.feature_names_in_)
    else:
        expected_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    st.sidebar.header("Entrer les caractéristiques")

    input_dict = {}
    cols = st.sidebar.columns(3)

    for i, col_name in enumerate(expected_columns):
        col = cols[i % 3]
        input_dict[col_name] = col.number_input(
            col_name,
            value=0.0,
            step=0.01
        )

    if st.sidebar.button("Prédire"):

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[expected_columns]

        input_scaled = scaler.transform(input_df)

        prediction = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 Transaction suspecte ! Probabilité : {proba:.2%}")
        else:
            st.success(f"✅ Transaction normale. Probabilité : {proba:.2%}")

# =============================
# ÉVALUATION
# =============================
elif section == "Évaluation":

    st.header("📈 Évaluation des modèles")

    rf_model, gb_model, scaler = load_models()
    df = load_data()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Alignement colonnes
    if hasattr(scaler, "feature_names_in_"):
        X = X[scaler.feature_names_in_]

    X_scaled = scaler.transform(X)

    # Random Forest
    y_pred_rf = rf_model.predict(X_scaled)
    y_score_rf = rf_model.predict_proba(X_scaled)[:, 1]

    st.subheader("Random Forest")
    st.write("Accuracy :", accuracy_score(y, y_pred_rf))
    st.write("F1-score :", f1_score(y, y_pred_rf))
    st.write("ROC-AUC :", roc_auc_score(y, y_score_rf))

    # Courbe ROC
    st.subheader("Courbe ROC")

    fpr_rf, tpr_rf, _ = roc_curve(y, y_score_rf)
    auc_rf = roc_auc_score(y, y_score_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")

    if gb_model is not None:
        y_score_gb = gb_model.predict_proba(X_scaled)[:, 1]
        fpr_gb, tpr_gb, _ = roc_curve(y, y_score_gb)
        auc_gb = roc_auc_score(y, y_score_gb)
        plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC={auc_gb:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()

    st.pyplot(plt)
