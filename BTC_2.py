import streamlit as st
from flask import Flask, jsonify, request, render_template
import numpy as np
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from scipy import stats
from statistics import mode
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
sys.modules['sklearn.externals.six'] = six


#st.write('Este es un predictor de tendencia de precios para los proximos minutos de BTC')

def dato_historico(symbol='tBTCUSD', timeframe='1m', limit=10000, section='hist'):

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

    # df.set_index('time',inplace=True)

    return df

# Agregamos los indicadores de cruces de medias y la ventana para para predecir sera de 60 minutos


def agregar_indicadores(df):
    cruces = [(2, 4), (4, 8), (8, 16), (15, 30),
              (20, 50), (50, 100), (50, 200)]

    # Agrego las medias
    for cruce in cruces:
        clave = str(cruce[0]) + '_' + str(cruce[1])
        df[clave] = (df.close.rolling(cruce[0]).mean() /
                     df.close.rolling(cruce[1]).mean() - 1)*100

    # Agrego el valor forward de 60 minutos
    df['fw_60'] = (df.close.shift(-20) / df.close - 1)*100
    df['pred'] = np.where(df.fw_60 > 0, 1, 0)

    df = df.dropna()

    return df


app = Flask('Servidor Get')


@app.route('/btcprice', methods=['GET'])
def hola():
    data = dato_historico()
    df_indicadores = data.copy()
    df_indicadores = agregar_indicadores(df_indicadores)
    df_indicadores.set_index('time', inplace=True)
    df_indicadores.drop(['open', 'close', 'high', 'low', 'volume', '2_4', '4_8', '8_16', '15_30',
                         '20_50', '50_100', '50_200', 'pred'], axis=1, inplace=True)
    adf_test = ADFTest(alpha=0.05)
    adf_test.should_diff(df_indicadores)
    df_train, df_test = train_test_split(
        df_indicadores, test_size=12, random_state=42, shuffle=False)
    arima_model = auto_arima(df_train, start_p=0, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0,
                             D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, seasonal=False, error_action='warn', trace=True,
                             supress_warnings=True, stepwise=True, random_state=20, n_fits=50)
    df_pred = pd.DataFrame(arima_model.predict(
        n_periods=12), index=df_test.index)
    df_pred.columns = ['pred']
    df_pred['pred'] = np.where(df_pred.pred > 0, 'Segun nuestro modelo hay un 76 porciento de probabilidades de que la tendencia sea alcista en los proximos 20m',
                               'Segun nuestro modelo hay un 76 porciento de probabilidades de que la tendencia sea a la baja en los proximos 20m')
    return(str(df_pred['pred'][0]))


@app.route('/', methods=['GET'])
def home():
    return render_template('BTC_2.html')


app.run(host='0.0.0.0',  port=5002)
