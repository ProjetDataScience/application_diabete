import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle et le scaler
model = joblib.load('model/diabete_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.title("Prédiction du diabète")

st.markdown("Entrez les données ci-dessous pour prédire le risque de diabète")

# Collecte des données utilisateur
Pregnancies = st.number_input('Nombre de grossesses', min_value=0, step=1)
Glucose = st.number_input('Taux de glucose', min_value=0)
BloodPressure = st.number_input('Pression artérielle (mg Hg)', min_value=0)
SkinThickness = st.number_input("Epaisseur de la peau (mm)", min_value=0)
Insulin = st.number_input("Insuline (mu U/ml)", min_value=0)
BMI = st.number_input("Indice de Masse Corporelle (IMC)", min_value=0.0, format="%.2f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
Age = st.number_input("Age", min_value=1, step=1)

# Créer un tableau de données avec l'entrée de l'utilisateur
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Exclure `pregnancies` de la normalisation
pregnancies_value = input_data[0, 0]
input_data_scaled = input_data[:, 1:]

# Normalisation des autres variables
input_data_scaled = scaler.transform(input_data_scaled)
# Réintégrer `Pregnancies` non normalisé dans les données transformées
input_data_scaled = np.column_stack([pregnancies_value, input_data_scaled])

# Bouton pour prédire
if st.button("🔍 Prédire") : 
    prediction = model.predict(input_data_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Risque de diabète détecté")
    if prediction == 0:
        st.success("✅ Aucun signe de diabète détecté")