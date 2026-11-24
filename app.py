import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Churn de Clientes")

# Formulario para introducir datos del cliente
st.header("Introduce los datos del cliente:")

contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
online_security = st.selectbox("Seguridad online", ["no", "yes", "no_internet_service"])
phoneservice = st.selectbox("Servicio telefónico", ["no", "yes"])
multiplelines = st.selectbox("Múltiples líneas", ["no", "no_phone_service", "yes"])
tech_support = st.selectbox("Soporte técnico", ["no", "yes", "no_internet_service"])
paperless_billing = st.selectbox("Facturación sin papel", ["no", "yes"])
senior_citizen = st.selectbox("¿Es persona mayor?", ["no", "yes"])
streaming_movies = st.selectbox("Películas en streaming", ["no", "yes", "no_internet_service"])
partner = st.selectbox("Pareja", ["no", "yes"])
online_backup = st.selectbox("Copia de seguridad online", ["no", "yes", "no_internet_service"])

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos del cliente
    client_data = {
        "contract": contract,
        "internetservice": internet_service,
        "onlinesecurity": online_security,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "techsupport": tech_support,
        "paperlessbilling": paperless_billing,
        "seniorcitizen": int(senior_citizen == "yes"),
        "streamingmovies": streaming_movies,
        "partner": partner,
        "onlinebackup": online_backup
    }

    # Transformar los datos del cliente
    X_client = dv.transform([client_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_client)[0][1]  # Probabilidad de churn

    # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")
