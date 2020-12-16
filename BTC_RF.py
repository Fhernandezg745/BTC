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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from pmdarima.arima import ADFTest
import pmdarima as pm
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import pytz
import time
import hashlib
import logging
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
import streamlit as st


st.write('Este es un predictor de tendencia de precios para los proximos minutos de BTC')

def dato_historico(moneda1='BTC', moneda2='USDT', timeframe='1m', desde=datetime.utcnow() - timedelta(weeks=24),
                   hasta='vacio',
                   limit=10000, section = 'hist'):
    
    start_time = time.time()


    logging.basicConfig(level=logging.INFO, format='{asctime} {levelname} ({threadName:11s}) {message}', style='{')
    print(f'Ticker {moneda1}')

    #Creo la variable Symbol
    if moneda2 == "USDT":
        moneda2 = "USD"
    if moneda1 == 'BCH':
        moneda1_ok = 'BCHN:'

    try:
        symbol='t'+moneda1_ok+moneda2
    except:
        symbol='t'+moneda1+moneda2

    # Presumo que las fechas son UTC0
    if hasta == 'vacio':
        hasta = datetime.now()
        hasta = hasta.replace(tzinfo=pytz.utc)+timedelta(minutes=1)
    else:
        #hasta = datetime.fromisoformat(hasta)
        hasta = hasta.replace(tzinfo=pytz.utc)+timedelta(days=1)

    desde = desde.replace(tzinfo=pytz.utc)

    # Llevo las variables Datetime a ms
    startTime = int(desde.timestamp() * 1000)
    endTime = int(hasta.timestamp() * 1000)

    # Inicializo el df acumulado
    df_acum = pd.DataFrame(columns=(0, 1, 2, 3, 4, 5))

    finished = False
    url = f'https://api-pub.bitfinex.com/v2/candles/trade:{timeframe}:{symbol}/{section}'

    contador = 0
    while not finished:

        try:
            ultimaFecha = df_acum.iloc[-1][0]
        except:
            ultimaFecha = startTime

        if (ultimaFecha >= endTime):
            break

        # Inicio Bajada
        logging.info(f'Bajada n° {contador}')
        params = {'limit': limit, 'start': ultimaFecha, 'end' : endTime, 'sort': '1'}

        r = requests.get(url, params=params)
        js = r.json()

        if js==[]:
            print(f'Problema con {moneda1}')
            finished=True

        # Armo el dataframe
        df = pd.DataFrame(js)

        # Verifico que traigo mas de una fila y es algo nuevo, si no, le doy break
        if len(df) ==1:
            try:
                if df.iloc[0][0] == df_acum.iloc[-1][0]:
                    break
            except:
                break

        df_acum = df_acum.append(df, sort=False)

        contador += 1

    # Convierto los valores strings a numeros
    df_acum = df_acum.apply(pd.to_numeric,errors='ignore')

    # Renombro las columnas segun lo acordado.
    df_acum.columns = ['time', 'open', 'close', 'high', 'low', 'volume']

    # Agrego columna ticker.
    df_acum['ticker'] = moneda1

    # Ordeno columnas segun lo acordado.
    df_acum = df_acum[['ticker','time', 'open', 'high', 'low', 'close', 'volume']]

    # Borro algún posible duplicado
    df_acum = df_acum.drop_duplicates(['time'], keep='last')

    # Elimino las filas que me trajo extras en caso que existan
    try:
        df_acum = df_acum[df_acum.time < hasta]
    except:
        pass

    # Paso a timestamp el time
    df_acum['time'] = pd.to_datetime(df_acum.time, unit='ms')

    # Le mando indice de time
    df_acum.set_index('time',inplace=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return df_acum
    
data = dato_historico()
# Guardamos la data
with open('btc_minutes.dat', 'wb') as file:
    pickle.dump(data,file)
# Levantamos la data
with open('btc_minutes.dat', 'rb') as file:
    data = pickle.load(file)
    
# Agregamos los indicadores de cruces de medias y la ventana para para predecir sera de 60 minutos

def agregar_indicadores(df):
    cruces = [(2,20),(2,40),(2,60),(2,100),(2,200),(5,20),(5,50),(5,100),(5,200),(5,400),(10,20),(10,50),(10,100),
             (10,200),(10,500),(20,50),(20,100),(20,200),(20,500),(20,1000),(50,100),(50,200),(50,500),(50,1000),
             (100,200),(100,400),(100,500),(100,1000),(200,500),(200,1000),(400,1000)]
    
    # Agrego las medias
    for cruce in cruces:
        clave = str(cruce[0]) + '_' + str(cruce[1])
        df[clave] = (df.close.rolling(cruce[0]).mean() / df.close.rolling(cruce[1]).mean() -1)*100
        
    # Agrego el valor forward de 20 minutos
    df['fw_20'] = (df.close.shift(-20) / df.close -1)*100
    df['pred'] = np.where(df.fw_20 > 0 ,1 ,0)
    
    df = df.dropna().round(4)
    
    return df

df_indicadores = data.copy()
df_indicadores = agregar_indicadores(df_indicadores)


# Primero seteamos df
df_forest = df_indicadores.copy()
df_forest.drop(['ticker','open','high','low','volume'], axis=1, inplace= True)

#split de train y test e instanciamiento de RF
X_train, X_test, y_train, y_test = train_test_split(df_forest.iloc[:,1:-2], df_forest.pred, test_size=0.2)
modelo_rf = RandomForestClassifier(criterion = 'entropy', max_depth=15)
modelo_rf.fit(X_train, y_train)
y_pred = modelo_rf.predict(X_test)
with open('bot_rf.dat', 'wb') as file:
    pickle.dump(modelo_rf,file)


m = np.array(skm.confusion_matrix(y_test, y_pred, normalize='all'))
#skm.plot_confusion_matrix(modelo_rf, X_test, y_test, normalize='all', cmap='Blues')

m = np.array(confusion_matrix(y_test, y_pred, normalize='all')).round(2)

matriz = {'true_positive':m[1][1], 'true_negative':m[0][0],'false_positive':m[0][1], 'false_negative':m[1][0]}

resumen = {'aciertos' : f"{matriz['true_positive']+matriz['true_negative']:.1%}", 
           'sesgo +': f"{matriz['false_positive']+matriz['false_negative']:.1%}"}

# Guardamos el modelo
with open('RD', 'wb') as file:
    pickle.dump(modelo_rf,file)

# levantar el modelo entrenado

def traerModelo(tipo='RF'):
    if tipo=='RF':
        with open('bot_rf.dat', 'rb') as file:
            modelo = pickle.load(file)
    else: 
        modelo = None
        print('no encontre el modelo que pediste')
    
    return modelo



# Traer data

def dato_historico_predecir(symbol ='tBTCUSD', timeframe = '1m', limit = 10000, section = 'hist'):
    
    # Endpoint
    url = f'https://api-pub.bitfinex.com/v2/candles/trade:{timeframe}:{symbol}/{section}'
    params = {'limit' : limit}
    
    # Pido la data
    r = requests.get(url, params = params)
    js = r.json()
    df = pd.DataFrame(js)
    
    # Convierto los valores strings a numeros
    df = df.apply(pd.to_numeric,errors='ignore')
    
    # Renombro las columnas.
    df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']
    
    # Paso a timestamp el time
    df['time'] = pd.to_datetime(df.time, unit='ms')
    
    # Ordeno la informacion para tener de lo mas viejo a lo mas nuevo
    df = df.sort_values(by ='time', ascending = True)
    
    df.set_index('time',inplace=True)
    
    df = df.dropna()

    return df



# Generar indicadores

def agregar_indicadores_predecir(df):
    cruces = [(2,20),(2,40),(2,60),(2,100),(2,200),(5,20),(5,50),(5,100),(5,200),(5,400),(10,20),(10,50),(10,100),
             (10,200),(10,500),(20,50),(20,100),(20,200),(20,500),(20,1000),(50,100),(50,200),(50,500),(50,1000),
             (100,200),(100,400),(100,500),(100,1000),(200,500),(200,1000),(400,1000)]
    
    # Agrego las medias
    for cruce in cruces:
        clave = str(cruce[0]) + '_' + str(cruce[1])
        df[clave] = (df.close.rolling(cruce[0]).mean() / df.close.rolling(cruce[1]).mean() -1)*100
    
    # Elimino columnas
    df.drop(['open', 'close', 'high', 'low', 'volume'],axis=1, inplace=True)
    
    df = df.dropna().round(4)
    
    return df


# PREDECIR
def predecir(data, modelo):
    try:
        actual = agregar_indicadores_predecir(data).iloc[-1]
        y_pred = modelo.predict((actual,))[0]
        y_proba = modelo.predict_proba((actual,))[0]
        return y_pred, y_proba
    except:
        print('No se pudo predecir')
        return None, None
    
modelo = traerModelo('RF')
data = dato_historico_predecir('tBTCUSD')

prediccion = predecir(data, modelo)


if st.button('Predecir Tendencia'):
    if  prediccion[0] == 0:
        print('\nPrediccion: en los proximos 20 minutos el BTC va a bajar con respecto al precio actual\n',prediccion[1])
    else:
        print('\nPrediccion: en los proximos 20 minutos el BTC va a subir con respecto al precio actual\n',prediccion[0])

    # Predecimos probabilidad resultado
    probabilidad = tuple(prediccion[1])
    print('Hora actual')
    print(datetime.now())
    print('\nPrediccion probabilidad\n',probabilidad)

    
    