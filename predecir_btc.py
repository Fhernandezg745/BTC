import streamlit as st
import logging
import hashlib
import time
import numpy as np
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from IPython.display import clear_output
import pickle
from scipy import stats
from statistics import mode
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import acf, pacf
import warnings
from bokeh.plotting import figure
warnings.filterwarnings('ignore')

# Funciones para predecir:
# levantar el modelo entrenado


def traerModelo(tipo='RF'):
    if tipo == 'RF':
        with open('bot_rf.pkl', 'rb') as file:
            modelo = pickle.load(file)
    else:
        modelo = None
        print('no encontre el modelo que pediste')

    return modelo


# Traer data
def dato_historico_predecir(symbol='tBTCUSD', timeframe='1m', limit=10000, section='hist'):

    # Endpoint
    url = f'https://api-pub.bitfinex.com/v2/candles/trade:{timeframe}:{symbol}/{section}'
    params = {'limit': limit}

    # Pido la data
    r = requests.get(url, params=params)
    js = r.json()
    df = pd.DataFrame(js)

    # Convierto los valores strings a numeros
    df = df.apply(pd.to_numeric, errors='ignore')

    # Renombro las columnas.
    df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']

    # Paso a timestamp el time
    df['time'] = pd.to_datetime(df.time, unit='ms')

    # Ordeno la informacion para tener de lo mas viejo a lo mas nuevo
    df = df.sort_values(by='time', ascending=True)

    df.set_index('time', inplace=True)

    df = df.dropna()

    return df


# Generar indicadores
def agregar_indicadores_predecir(df):
    cruces = [(2, 20), (2, 40), (2, 60), (2, 100), (2, 200), (5, 20), (5, 50), (5, 100), (5, 200), (5, 400), (10, 20), (10, 50), (10, 100),
              (10, 200), (10, 500), (20, 50), (20, 100), (20, 200), (20,
                                                                     500), (20, 1000), (50, 100), (50, 200), (50, 500), (50, 1000),
              (100, 200), (100, 400), (100, 500), (100, 1000), (200, 500), (200, 1000), (400, 1000)]

    # Agrego las medias
    for cruce in cruces:
        clave = str(cruce[0]) + '_' + str(cruce[1])
        df[clave] = (df.close.rolling(cruce[0]).mean() /
                     df.close.rolling(cruce[1]).mean() - 1)*100

    # Elimino columnas
    df.drop(['open', 'close', 'high', 'low', 'volume'], axis=1, inplace=True)

    df = df.dropna().round(4)

    return df


# PREDECIR
def predecir(data, modelo):
    try:
        actual = agregar_indicadores_predecir(data).iloc[-1]
        y_pred = modelo.predict((actual,))[0]
        # pruebo con accuracy
        # y_proba = accuracy_score(y_test, modelo.predict(X_test))
        y_proba = modelo.predict_proba((actual,))[0]

        return y_pred, y_proba
    except:
        print('No se pudo predecir')
        return None, None


# Predecimos probabilidad resultado
# probabilidad = tuple(prediccion[1])
# print('\nPrediccion probabilidad\n', probabilidad)
data_grafico = dato_historico_predecir('tBTCUSD')
data_grafico['time'] = data_grafico.index
x = data_grafico['time']
y = data_grafico['close']
p = figure(
    title='evolucion del precio',
    x_axis_label='hora',
    y_axis_label='precio de cierre')
p.line(x, y, legend_label='Trend', line_width=2)
st.bokeh_chart(p, use_container_width=True)

if st.button('Predecir Bitcoin ahora'):
    # Cuando se hace click al boton se ejecuta esta secuencia
    modelo = traerModelo('RF')
    data_predecir = dato_historico_predecir('tBTCUSD')
    prediccion = predecir(data_predecir, modelo)
    data_grafico = dato_historico_predecir('tBTCUSD')
    # fig = go.Figure(data=[go.Candlestick(x=data_grafico['time'], open=data_grafico['open'],
    #                                     high=data_grafico['high'], low=data_grafico['low'], close=data_grafico['close'])])
    # imprimimos el horario
    # st.write('Hora actual', datetime.now())
    if prediccion[0] == 0:
        st.write('Hora actual', datetime.now(),
                 '\nPrediccion: en los proximos 10 minutos el BTC va a bajar con respecto al precio actual\n', '1 BTC = ',
                 data_grafico['close'][-1], "USD")
    else:
        st.write('Hora actual', datetime.now(),
                 '\nPrediccion: en los proximos 10 minutos el BTC va a subir con respecto al precio actual\n', '1 BTC = ',
                 data_grafico['close'][-1], "USD")
