import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web

#print("Definisci il limite commissionale annuo")
#Spesa_massima_annua=input()

data_inizio_analisi= dt.datetime(2019,12, 31)
data_fine_analisi= dt.datetime (2020, 12, 31)

# print("Quale è il primo asset da inserire in analisi")
# Ticker1 = input()
#
# print("Quale è il secondo asset da inserire in analisi")
# Ticker2 = input()
#
# print("Quale è il terzo asset da inserire in analisi")
# Ticker3 = input()
#
# print("Quale è il quarto asset da inserire in analisi")
# Ticker4 = input()
#
# print("Quale è il quinto asset da inserire in analisi")
# Ticker5 = input()
#
# dataset_Asset=[Ticker1, Ticker2, Ticker3, Ticker4, Ticker5]
dataset_Asset=['KO', 'TSLA', 'NIO', 'AAPL', 'GM']
print(dataset_Asset)

dati_Asset_1 = web.DataReader( dataset_Asset[0], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[0], dati_Asset_1.shape)
dati_Asset_2 = web.DataReader( dataset_Asset[1], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[1], dati_Asset_2.shape)
dati_Asset_3 = web.DataReader( dataset_Asset[2], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[2], dati_Asset_3.shape)
dati_Asset_4 = web.DataReader( dataset_Asset[3], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[3], dati_Asset_4.shape)
dati_Asset_5 = web.DataReader( dataset_Asset[4], "yahoo", data_inizio_analisi,  data_fine_analisi)
print(dataset_Asset[4], dati_Asset_5.shape)

quotazioni_1 = dati_Asset_1.values
lista_chiusura_Asset_1 = quotazioni_1 [:,5]
quotazioni_2 = dati_Asset_2.values
lista_chiusura_Asset_2 = quotazioni_2 [:,5]
quotazioni_3 = dati_Asset_1.values
lista_chiusura_Asset_3 = quotazioni_3 [:,5]
quotazioni_4 = dati_Asset_1.values
lista_chiusura_Asset_4 = quotazioni_4 [:,5]
quotazioni_5 = dati_Asset_1.values
lista_chiusura_Asset_5 = quotazioni_5 [:,5]

lista_rendimenti_Asset_1= np.array([])
lista_rendimenti_Asset_2= np.array([])
lista_rendimenti_Asset_3= np.array([])
lista_rendimenti_Asset_4= np.array([])
lista_rendimenti_Asset_5= np.array([])

i=0
lunghezza = lista_chiusura_Asset_1.size
while (i<lunghezza-1):
    rendimento=np.log (lista_chiusura_Asset_1[i+1]/lista_chiusura_Asset_1[i])
    lista_rendimenti_Asset_1 = np.append (lista_rendimenti_Asset_1, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_2[i+1]/lista_chiusura_Asset_2[i])
    lista_rendimenti_Asset_2 = np.append (lista_rendimenti_Asset_2, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_3[i+1]/lista_chiusura_Asset_3[i])
    lista_rendimenti_Asset_3 = np.append (lista_rendimenti_Asset_3, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_4[i+1]/lista_chiusura_Asset_4[i])
    lista_rendimenti_Asset_4 = np.append (lista_rendimenti_Asset_4, [rendimento])

    rendimento=np.log (lista_chiusura_Asset_5[i+1]/lista_chiusura_Asset_5[i])
    lista_rendimenti_Asset_5 = np.append (lista_rendimenti_Asset_5, [rendimento])

    i=i+1
